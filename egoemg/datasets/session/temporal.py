"""Cached frames around an unchanged dense30/sparse center sample."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from egoemg.datasets.session.schema import _array
from egoemg.datasets.session.fullframe import EgoEmgSessionFullFrameDataset
from egoemg.datasets.session.alignment import dependency_signature, ego_stream
from egoemg.datasets.evaluation_partition import EvaluationPartition


def temporal_neighbors(
    frames, times, allowed, centers, step, num_frames=5, causal=False
):
    """Nearest cached native ordinals, without jumping over partition/time gaps."""
    if num_frames < 1 or num_frames % 2 != 1 or step <= 0:
        raise ValueError("Expected an odd positive frame count and positive step")
    middle = num_frames - 1 if causal else num_frames // 2
    frames = np.asarray(frames)
    if not len(frames):
        raise ValueError("Empty temporal frame cache")
    gaps = np.diff(times)
    cut = (allowed[1:] != allowed[:-1]) | (gaps > 250_000) | (gaps <= 0)
    segments = np.cumsum(np.r_[True, cut])
    offsets = (
        np.arange(-(num_frames - 1), 1) if causal else np.arange(-middle, middle + 1)
    )
    desired = centers[:, None] + offsets[None, :] * step
    right = np.searchsorted(frames, desired).clip(0, len(frames) - 1)
    left = (right - 1).clip(0, len(frames) - 1)
    rows = np.where(
        abs(frames[left] - desired) <= abs(frames[right] - desired), left, right
    )
    if causal:
        if np.any(np.diff(frames) < 0):
            raise ValueError("Causal context coordinates must be monotonic")
        rows = (np.searchsorted(frames, desired, side="right") - 1).clip(
            0, len(frames) - 1
        )
    anchor = rows[:, middle]
    valid = (
        (desired >= frames[0])
        & (desired <= frames[-1])
        & (abs(frames[rows] - desired) <= max(2, step // 4))
        & allowed[rows]
        & allowed[anchor, None]
        & (segments[rows] == segments[anchor, None])
    )
    rows = np.where(valid, rows, anchor[:, None]).astype(np.int32)
    valid[:, middle] = True
    return rows, valid


class EgoEmgSessionTemporalVisionDataset(EgoEmgSessionFullFrameDataset):
    """Keep center identities/labels; add only in-partition cached RGB context."""

    def __init__(
        self,
        *args,
        context_cache_dir=None,
        context_partition_path=None,
        num_frames=5,
        frame_interval_us=None,
        causal_context=False,
        **kwargs,
    ):
        if num_frames < 1 or num_frames % 2 != 1:
            raise ValueError("num_frames must be positive and odd")
        if frame_interval_us is not None and frame_interval_us <= 0:
            raise ValueError("frame_interval_us must be positive")
        self.num_frames = int(num_frames)
        self.causal_context = bool(causal_context)
        self.middle = (
            self.num_frames - 1 if self.causal_context else self.num_frames // 2
        )
        super().__init__(*args, **kwargs)
        context_dir = Path(
            context_cache_dir or self.dataset_dir / "fullframe_256_30fps"
        )
        self._context_rgb = OrderedDict()
        self._context = []
        self.clip_rows = np.empty((len(self), self.num_frames), dtype=np.int32)
        self.clip_valid = np.empty((len(self), self.num_frames), dtype=bool)
        policy = (
            EvaluationPartition(context_partition_path, "val")
            if context_partition_path is not None
            else self.evaluation_partition
        )
        wanted_role = (
            self.evaluation_partition.code
            if self.evaluation_partition is not None
            else 0
        )
        for record_id, record in enumerate(self.records):
            session, folder = record["session"], record["folder"]
            stem = context_dir / session["session_id"]
            meta = stem.with_suffix(".json")
            if not meta.is_file():
                raise RuntimeError(f"Prepare dense30 context cache first: {stem}")
            signature = dependency_signature(folder / "session.json", session, 0, 30)
            if json.loads(meta.read_text())["signature"] != signature:
                raise RuntimeError(f"Stale dense30 context cache: {stem}")
            if stem == record["stem"]:
                alignment = record["alignment"]
            else:
                with np.load(stem.with_suffix(".npz")) as data:
                    alignment = {k: data[k].copy() for k in data.files}
            roles = alignment["splits"].copy()
            if policy is not None:
                roles[:] = -1
                for h, side in enumerate(("left", "right")):
                    hand = session["hands"].get(side)
                    if not hand:
                        continue
                    arrays = hand["arrays"]
                    rows = alignment["rows"][:, h]
                    gestures = (
                        _array(folder, arrays["gesture_class"])[rows]
                        if "gesture_class" in arrays
                        else np.full(len(rows), -1)
                    )
                    objects = (
                        _array(folder, arrays["object_id"])[rows]
                        if "object_id" in arrays
                        else np.full(len(rows), -1)
                    )
                    roles[:, h] = policy.classify(
                        session["subject"],
                        alignment["splits"][:, h],
                        gestures,
                        objects,
                        session_id=session["session_id"],
                    )
            allowed = (roles == wanted_role).any(1) & ~(
                (roles >= 0) & (roles != wanted_role)
            ).any(1)
            raw_times = _array(folder, ego_stream(session)["arrays"]["timestamps"])
            dt = np.diff(raw_times)
            dt = dt[dt > 0]
            fps = 1_000_000 / np.median(dt) if len(dt) else 30.0
            step = 40 if fps >= 45 else 30
            positions = np.flatnonzero(self.samples[:, 0] == record_id)
            local = self.samples[positions, 1]
            centers = record["alignment"]["frames"][local]
            if frame_interval_us is None:
                coordinates = alignment["frames"]
                query = centers
                spacing = step
            else:
                coordinates = alignment["times"].astype(np.int64)
                query = record["alignment"]["times"][local].astype(np.int64)
                spacing = int(frame_interval_us)
                step = max(1, round(fps * spacing / 1_000_000))
            clip_rows, valid = temporal_neighbors(
                coordinates,
                alignment["times"],
                allowed,
                query,
                spacing,
                self.num_frames,
                causal=self.causal_context,
            )
            self.clip_rows[positions] = clip_rows
            self.clip_valid[positions] = valid
            self._context.append(
                dict(
                    stem=stem,
                    frames=alignment["frames"],
                    times=alignment["times"],
                    native_fps=float(fps),
                    step=step,
                )
            )
        self.batch_read_signature = ("temporal_fullframe_bimanual_256", self.num_frames)

    def _read_into(self, index, buffers=None):
        # Keep the exact original center RGB, including sparse val/test centers
        # that may not be present verbatim in the 30fps context cache.
        result = super()._read_into(index)
        center = result["vision_img"]
        record_id = int(self.samples[index, 0])
        info = self._context[record_id]
        if record_id not in self._context_rgb:
            if info["stem"] == self.records[record_id]["stem"]:
                self._context_rgb[record_id] = self._cache[record_id]["rgb"]
            else:
                self._context_rgb[record_id] = np.load(
                    info["stem"].with_suffix(".npy"), mmap_mode="r"
                )
            while len(self._context_rgb) > self.max_open_sessions:
                self._context_rgb.popitem(last=False)
        self._context_rgb.move_to_end(record_id)
        rows, valid = self.clip_rows[index], self.clip_valid[index]
        images = (
            buffers["vision_img"]
            if buffers is not None
            else np.empty((self.num_frames, *center.shape), dtype=center.dtype)
        )
        for position, row in enumerate(rows):
            if position == self.middle or not valid[position]:
                images[position] = center
            else:
                images[position] = self._context_rgb[record_id][row].transpose(2, 0, 1)
        frame_ids = info["frames"][rows].astype(np.int64).copy()
        frame_ids[~valid] = result["camera_frame"]
        frame_ids[self.middle] = result["camera_frame"]
        frame_times = info["times"][rows].astype(np.int64).copy()
        center_time = self.records[record_id]["alignment"]["times"][
            self.samples[index, 1]
        ]
        frame_times[~valid] = center_time
        frame_times[self.middle] = center_time
        if self.causal_context and np.any(frame_times > center_time):
            raise ValueError("Future RGB entered a causal context")
        result.update(
            vision_img=images,
            temporal_valid_mask=valid.copy(),
            temporal_camera_frames=frame_ids,
            temporal_timestamps_us=frame_times,
            temporal_step_frames=np.int64(info["step"]),
        )
        if buffers is not None:
            for key, buffer in buffers.items():
                buffer[:] = result[key]
                result[key] = buffer
        return result
