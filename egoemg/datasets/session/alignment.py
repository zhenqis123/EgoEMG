"""Camera clocks, native frame alignment and cache signatures."""

import numpy as np
from egoemg.datasets.session.schema import SPLIT_IDS, _array, _valid_label_key


def align_camera_rows(hand_times, camera_times, maximum_age_us=50_000):
    """Use the preceding hand sample, matching the published crop geometry."""
    order = None
    if np.any(np.diff(hand_times) < 0):
        order = np.argsort(hand_times, kind="stable")
        hand_times = hand_times[order]
    positions = np.searchsorted(hand_times, camera_times, side="right") - 1
    safe = np.clip(positions, 0, len(hand_times) - 1)
    ages = camera_times - hand_times[safe]
    valid = (positions >= 0) & (ages >= 0) & (ages <= maximum_age_us)
    return (order[safe] if order is not None else safe), valid, ages


def ego_stream(session):
    cameras = session["cameras"]
    device = "showee_head" if cameras.get("showee_head") else "gopro"
    name = "camera" if device == "showee_head" else "rgb"
    return cameras[device]["streams"][name]


def dependency_signature(path, session, interval_us, target_fps=None):
    stream = ego_stream(session)
    files = [path, path.parent / stream["video"]["path"]]
    files.extend(path.parent / v["path"] for v in stream["arrays"].values())
    video = stream["video"]
    if video.get("frame_index_map"):
        files.append(path.parent / video["frame_index_map"]["path"])
    for hand in session["hands"].values():
        if hand:
            for key in (
                "timestamps",
                "split_id",
                "joint_angles",
                "joint_angles_valid",
                "label_valid",
                "wrist_angles",
                "wrist_valid",
                "object_id",
            ):
                if key in hand["arrays"]:
                    files.append(path.parent / hand["arrays"][key]["path"])
    result = dict(
        version=1,
        interval_us=interval_us,
        files={str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files},
    )
    if target_fps is not None:
        result.update(version=2, target_fps=target_fps)
    return result


def build_alignment(folder, session, interval_us=400_000, target_fps=None):
    stream = ego_stream(session)
    times = _array(folder, stream["arrays"]["timestamps"])
    n = len(times)
    rows = np.zeros((n, 2), np.int64)
    splits = np.full((n, 2), -1, np.int8)
    camera_valid = np.ones(n, bool)
    if "timestamp_valid" in stream["arrays"]:
        camera_valid &= _array(folder, stream["arrays"]["timestamp_valid"]).astype(bool)
    for index, name in enumerate(("left", "right")):
        hand = session["hands"].get(name)
        if not hand:
            continue
        arrays = hand["arrays"]
        aligned, valid, _ = align_camera_rows(
            _array(folder, arrays["timestamps"]), times
        )
        rows[:, index] = aligned
        valid &= camera_valid
        valid &= _array(folder, arrays[_valid_label_key(arrays)])[aligned].astype(bool)
        valid &= np.isfinite(_array(folder, arrays["joint_angles"])[aligned]).all(1)
        values = _array(folder, arrays["split_id"])[aligned]
        valid &= np.isin(values, list(SPLIT_IDS.values()))
        splits[valid, index] = values[valid]
    frames = np.flatnonzero((splits >= 0).any(1))
    if target_fps is not None and (target_fps <= 0 or interval_us):
        raise ValueError("Target FPS requires positive FPS and interval_us=0")
    if len(frames) and (interval_us or target_fps is not None):
        if target_fps is None:
            bins = times[frames] // interval_us
        else:
            # Integer arithmetic avoids rounding large absolute timestamps.
            t = times[frames]
            bins = (t // 1_000_000) * target_fps + (
                (t % 1_000_000) * target_fps
            ) // 1_000_000
        # Preserve split/task transitions even if they occur within one time bin.
        changed = np.any(splits[frames][1:] != splits[frames][:-1], axis=1)
        frames = frames[np.r_[True, (bins[1:] != bins[:-1]) | changed]]
    video = stream["video"]
    physical = (
        _array(folder, video["frame_index_map"])[frames]
        if video.get("frame_index_map")
        else frames
    )
    if np.any(physical < 0):
        raise ValueError("Negative physical camera ordinal")
    return dict(
        frames=frames,
        physical=physical,
        rows=rows[frames],
        splits=splits[frames],
        times=times[frames],
    )
