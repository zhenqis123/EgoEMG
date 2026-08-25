"""Compute session-continuous frame indices for ShowEE wrist and ZED views.

ShowEE sessions record three additional video streams per action:

  * ``showee_left_wrist`` / ``showee_right_wrist``: per-action MKVs with
    per-action timestamp TXT files (same format as the head view), so the
    local frame index is ``nearest(txt_ts, target_us)``.
  * ``zed_rgbd/rgb.mkv``: per-action video whose frames correspond 1:1
    (by order) to the jsonl timestamp entries inside the action's time
    window (the ZED logs extra pre/post-action entries that the video
    does not contain).

The released videos are session-level concatenations, so this script
writes session-continuous indices (local + cumulative mkv frame counts,
skipping actions whose recording is missing — exactly the skip order of
``build_showee_session_videos.py``) into new memmap fields::

    image_wrist_left_frame_index/stale/delta_ms
    image_wrist_right_frame_index/stale/delta_ms
    image_zed_frame_index                      (rewritten; was jsonl-global)

Usage::

    python scripts/prepare/build_showee_wrist_zed_indices.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_full_memmap \
        --source-memmap-dir /data/xiziheng/EgoEMG_full_memmap \
        --showee-root /mnt/nvme/xiziheng/showee/downloads \
        --showee-root /mnt/nvme/xiziheng \
        --allintra-root /mnt/nvme/xiziheng/EgoEMG_videos
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

STREAM_CONF = {
    "wrist_left": {"dir": "showee_left_wrist", "mkv": "*.mkv", "ts": "txt"},
    "wrist_right": {"dir": "showee_right_wrist", "mkv": "*.mkv", "ts": "txt"},
    "zed": {"dir": "zed_rgbd", "mkv": "rgb.mkv", "ts": "jsonl"},
}


def _clean(values) -> list[str]:
    out = []
    for v in values:
        s = v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
        s = s.strip("b'").strip('"')
        out.append(s)
    return out


def _find_session_dir(roots: list[Path], name: str) -> Path | None:
    for root in roots:
        p = root / name
        if p.is_dir():
            return p
    return None


def _mkv_frames(path: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip() and r.stdout.strip() != "N/A":
        return int(r.stdout.strip())
    # Some ZED containers lack the nb_frames header.  Try duration x fps
    # (CFR streams; exact up to rounding), then fall back to decord.
    r2 = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration,avg_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    fields = r2.stdout.strip().split(",")
    fps = next((f for f in fields if "/" in f), None)
    dur = next((f for f in fields if f != "N/A" and "/" not in f), None)
    if fps is not None and dur is not None:
        num, den = fps.split("/")
        return int(round(float(dur) * int(num) / int(den)))
    import ctypes
    ctypes.CDLL(
        "/home/xiziheng/miniconda3/envs/emg2pose_env/lib/libstdc++.so.6",
        mode=ctypes.RTLD_GLOBAL)
    from decord import VideoReader, cpu
    return len(VideoReader(str(path), ctx=cpu(0)))


def _read_txt_ts(txt_path: Path) -> np.ndarray:
    data = np.loadtxt(txt_path, delimiter=",", dtype=np.int64, ndmin=2)
    values = data[:, 1]
    values = values // 1000 if np.median(values) > 10 ** 17 else values
    return values


def _read_zed_ts(jsonl_path: Path) -> np.ndarray:
    rows: list[int] = []
    with jsonl_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(int(json.loads(line)["timestamp"]))
    return np.asarray(rows, dtype=np.int64)


def _nearest_indices(source_us: np.ndarray, target_us: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_us, target_us, side="left")
    right = np.clip(right, 0, len(source_us) - 1)
    left = np.clip(right - 1, 0, len(source_us) - 1)
    choose_left = np.abs(target_us - source_us[left]) <= np.abs(
        source_us[right] - target_us)
    return np.where(choose_left, left, right).astype(np.int64)


def _local_indices(stream: str, action_dir: Path, target_us: np.ndarray,
                   action_start_us: int, action_end_us: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Per-action local video frame indices + |timestamp| delta in ms.

    Returns (indices, delta_ms); unavailable rows are -1 / int32 max.
    """
    conf = STREAM_CONF[stream]
    stream_dir = action_dir / conf["dir"]
    n = len(target_us)
    bad = (np.full(n, -1, dtype=np.int32),
           np.full(n, np.iinfo(np.int32).max, dtype=np.int32))
    if not stream_dir.is_dir():
        return bad
    if conf["ts"] == "txt":
        txts = sorted(stream_dir.glob("*.txt"))
        if len(txts) != 1:
            return bad
        ts = _read_txt_ts(txts[0])
    else:  # jsonl
        jsonl = stream_dir / "rgb_timestamps.jsonl"
        if not jsonl.is_file():
            return bad
        ts = _read_zed_ts(jsonl)
        # The video covers only the jsonl entries inside the action window,
        # 1:1 by order.  The local frame index is the position of the
        # nearest entry within the window.
        margin = 100_000  # 100 ms
        in_win = (ts >= action_start_us - margin) & (ts <= action_end_us + margin)
        if not in_win.any():
            return bad
        win_idx = np.nonzero(in_win)[0]
        order = np.searchsorted(ts[win_idx], target_us, side="left")
        order = np.clip(order, 0, len(win_idx) - 1)
        left = np.clip(order - 1, 0, len(win_idx) - 1)
        nearest = np.where(
            np.abs(target_us - ts[win_idx][left])
            <= np.abs(ts[win_idx][order] - target_us),
            left, order)
        nearest = np.clip(nearest, 0, len(win_idx) - 1)
        delta_us = np.abs(target_us - ts[win_idx][nearest])
        idx = nearest.astype(np.int32)
        out = np.where(delta_us <= 100_000, idx, -1).astype(np.int32)
        dms = np.where(delta_us <= 100_000, np.rint(delta_us / 1000.0).astype(np.int32),
                       np.iinfo(np.int32).max)
        return out, dms
    if len(ts) == 0:
        return bad
    idx = _nearest_indices(ts, target_us)
    delta_us = np.abs(ts[idx] - target_us)
    out = np.where(delta_us <= 100_000, idx, -1).astype(np.int32)
    dms = np.where(delta_us <= 100_000, np.rint(delta_us / 1000.0).astype(np.int32),
                   np.iinfo(np.int32).max)
    return out, dms


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--source-memmap-dir", type=Path, required=True,
                    help="Memmap holding the original per-action (928-episode) "
                         "metadata for exact action boundaries.")
    ap.add_argument("--source-metadata", type=Path, default=None,
                    help="Path to the 928-episode metadata.npz (defaults to "
                         "<source-memmap-dir>/metadata.npz.orig928 if present, "
                         "else metadata.npz)")
    ap.add_argument("--showee-root", type=Path, nargs="+", required=True)
    ap.add_argument("--allintra-root", type=Path, required=True)
    ap.add_argument("--corrupt-list", type=Path, default=None,
                    help="JSON list of source MKV paths that were excluded "
                         "from the session videos (unfinalized recordings)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    corrupt: set[str] = set()
    if args.corrupt_list is not None:
        corrupt = set(json.loads(args.corrupt_list.read_text()))

    target = np.load(args.memmap_dir / "metadata.npz", allow_pickle=False)
    with (args.memmap_dir / "manifest.json").open() as f:
        manifest = json.load(f)
    src_path = args.source_metadata
    if src_path is None:
        alt = args.source_memmap_dir / "metadata.npz.orig928"
        src_path = alt if alt.is_file() else args.source_memmap_dir / "metadata.npz"
    src = np.load(src_path, allow_pickle=False)
    print(f"source metadata: {src_path} ({len(src['episode_id'])} episodes)")
    target_parquet = _clean(target["episode_source_parquet"])
    src_parquet = _clean(src["episode_source_parquet"])
    src_start = src["episode_start_idx"].astype(np.int64)
    src_end = src["episode_end_idx"].astype(np.int64)

    ts_mm = np.memmap(args.memmap_dir / "timestamp_us.dat",
                      dtype=np.int64, mode="r",
                      shape=(int(manifest["total_rows"]),))

    n_rows = int(manifest["total_rows"])
    mmaps = {}
    for stream in STREAM_CONF:
        for suffix in ("frame_index", "stale", "delta_ms"):
            name = f"image_{stream}_{suffix}"
            if name in manifest["fields"]:
                mmaps[name] = np.memmap(
                    args.memmap_dir / manifest["fields"][name]["filename"],
                    dtype=manifest["fields"][name]["dtype"], mode="r+",
                    shape=tuple(manifest["fields"][name]["shape"]))
            else:
                if args.dry_run:
                    mmaps[name] = None
                    continue
                dtype = "int32" if suffix != "stale" else "bool"
                info = {"filename": f"{name}.dat", "dtype": dtype,
                        "shape": [n_rows]}
                manifest["fields"][name] = info
                mm = np.memmap(args.memmap_dir / f"{name}.dat", mode="w+",
                               dtype=dtype, shape=(n_rows,))
                mm[:] = (-1 if suffix != "stale" else True)
                mm.flush()
                mmaps[name] = mm
                print(f"created field {name}")

    session_eps = [
        i for i, p in enumerate(target_parquet)
        if p and _find_session_dir(args.showee_root, p) is not None
    ]
    print(f"{len(session_eps)} ShowEE sessions")

    report = {}
    for ep_idx in session_eps:
        session = target_parquet[ep_idx]
        s = int(target["episode_start_idx"][ep_idx])
        e = int(target["episode_end_idx"][ep_idx])
        # Legacy metadata stored the LAST row (inclusive); regenerated
        # metadata uses the exclusive bound.
        if e == s + int(target["episode_length"][ep_idx]) - 1:
            e += 1
        session_dir = _find_session_dir(args.showee_root, session)
        assert session_dir is not None

        actions = sorted(
            [i for i, p in enumerate(src_parquet) if p.startswith(session + "/")],
            key=lambda i: int(src_start[i]))
        ranges = [(int(src_start[a]), int(src_end[a])) for a in actions]
        if ranges[0][0] != s or ranges[-1][1] != e:
            report[session] = {"error": "range mismatch"}
            print(f"  {session}: range mismatch, skipping")
            continue

        # Session video frame counts (the built session videos).
        video_frames = {}
        for stream in STREAM_CONF:
            vp = args.allintra_root / \
                f"episode_{ep_idx:06d}_{stream}_allintra.mp4"
            if not vp.is_file():
                report[session] = {"error": f"missing {vp.name}"}
                print(f"  {session}: missing {vp.name}, skipping")
                break
            video_frames[stream] = _mkv_frames(vp)
        else:
            cum = {st: 0 for st in STREAM_CONF}
            for a, (a_s, a_e) in zip(actions, ranges):
                a_name = src_parquet[a].split("/")[1]
                a_dir = session_dir / a_name
                if not a_dir.is_dir():
                    # thum -> thumb alias like the head-view pipeline.
                    if a_name == "thum" and (session_dir / "thumb").is_dir():
                        a_dir = session_dir / "thumb"
                    else:
                        a_dir = None
                target_us = np.asarray(ts_mm[a_s:a_e])
                for stream in STREAM_CONF:
                    conf = STREAM_CONF[stream]
                    n_mkv = 0
                    mkv_path = None
                    if a_dir is not None:
                        mkvs = sorted(
                            (a_dir / conf["dir"]).glob(conf["mkv"])) \
                            if (a_dir / conf["dir"]).is_dir() else []
                        if len(mkvs) == 1:
                            mkv_path = mkvs[0]
                            if str(mkv_path) not in corrupt:
                                n_mkv = _mkv_frames(mkv_path)
                    if a_dir is None or mkv_path is None or str(mkv_path) in corrupt:
                        local = np.full(len(target_us), -1, dtype=np.int32)
                        delta_ms = np.full(
                            len(target_us), np.iinfo(np.int32).max, dtype=np.int32)
                    else:
                        local, delta_ms = _local_indices(
                            stream, a_dir, target_us,
                            int(target_us[0]), int(target_us[-1]))
                        # Some timestamp files have more entries than the
                        # video (dropped/unsaved frames): clamp local indices
                        # to the recording's last frame so they stay inside
                        # the action's segment of the session video.
                        local = np.where(
                            local >= 0, np.minimum(local, n_mkv - 1), -1)
                    # The session video concatenates every action that HAS a
                    # decodable recording, so the cumulative offset advances by
                    # its mkv frame count regardless of memmap row coverage.
                    # Unfinalized (corrupt) recordings were excluded from the
                    # session videos and their rows are marked stale.
                    if not args.dry_run:
                        fi = mmaps[f"image_{stream}_frame_index"]
                        st = mmaps[f"image_{stream}_stale"]
                        dm = mmaps[f"image_{stream}_delta_ms"]
                        fi[a_s:a_e] = np.where(local >= 0, local + cum[stream], -1)
                        st[a_s:a_e] = local < 0
                        dm[a_s:a_e] = delta_ms
                    cum[stream] += n_mkv
            report[session] = {
                "actions": len(actions),
                "video_frames": video_frames,
                "cumulative_frames": cum,
                "ok": all(video_frames[st] == cum[st] for st in STREAM_CONF),
            }
            flag = "OK" if report[session]["ok"] else "MISMATCH"
            print(f"  {session}: video={video_frames} cum={cum} {flag}")

    for mm in mmaps.values():
        if mm is not None:
            mm.flush()
    if not args.dry_run:
        shutil.copy2(args.memmap_dir / "manifest.json",
                     args.memmap_dir / "manifest.json.bak2")
        (args.memmap_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2))
    with (args.memmap_dir / "wrist_zed_indices_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
