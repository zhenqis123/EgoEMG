"""Build session-level ShowEE videos for the wrist and ZED streams.

The released head-view ShowEE videos are session-level concatenations
(one video per session, ~40 action recordings joined end-to-end).  This
script builds the same for the other recorded streams:

  * ``showee_left_wrist``  -> ``episode_XXXXXX_wrist_left_allintra.mp4``
  * ``showee_right_wrist`` -> ``episode_XXXXXX_wrist_right_allintra.mp4``
  * ``zed_rgbd/rgb.mkv``   -> ``episode_XXXXXX_zed_allintra.mp4``

Per-action recordings are concatenated (concat demuxer, identical
recorder codecs) and re-encoded once with the same all-intra H.264
settings as the head-view pipeline.  Actions whose recording is missing
for a stream are skipped; the frame-index rewrite must use the same
skip order (missing actions contribute zero frames).

Usage::

    python scripts/prepare/build_showee_session_videos.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_full_memmap \
        --showee-root /mnt/nvme/xiziheng/showee/downloads \
        --showee-root /mnt/nvme/xiziheng \
        --allintra-root /mnt/nvme/xiziheng/EgoEMG_videos \
        --streams wrist_left wrist_right zed \
        --jobs 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STREAM_DIRS = {
    "wrist_left": "showee_left_wrist",
    "wrist_right": "showee_right_wrist",
    "zed": "zed_rgbd",
}
STREAM_MKV = {
    "wrist_left": "*.mkv",
    "wrist_right": "*.mkv",
    "zed": "rgb.mkv",
}

ENCODE_ARGS = [
    "-an", "-c:v", "libx264",
    "-preset", "ultrafast", "-crf", "23",
    "-tune", "fastdecode",
    "-x264-params", "keyint=1:min-keyint=1:scenecut=0:bframes=0:ref=1:cabac=0",
    "-movflags", "+faststart",
]


def _clean(values) -> list[str]:
    import numpy as np

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
        "/home/xiziheng/miniconda3/envs/egoemg_env/lib/libstdc++.so.6",
        mode=ctypes.RTLD_GLOBAL)
    from decord import VideoReader, cpu
    return len(VideoReader(str(path), ctx=cpu(0)))


def _action_order(session_dir: Path) -> list[str]:
    """Sorted action dir names (matches the memmap's per-action order)."""
    return sorted(p.name for p in session_dir.iterdir() if p.is_dir())


def build_one(session_name: str, stream: str, roots: list[Path],
              out_root: Path, ep_idx: int,
              corrupt: set[str] | None = None,
              verify_only: bool = False) -> dict:
    session_dir = _find_session_dir(roots, session_name)
    assert session_dir is not None
    stream_dir_name = STREAM_DIRS[stream]
    mkv_glob = STREAM_MKV[stream]
    corrupt = corrupt or set()

    sources: list[Path] = []
    skipped: list[str] = []
    for action in _action_order(session_dir):
        sd = session_dir / action / stream_dir_name
        mkvs = sorted(sd.glob(mkv_glob)) if sd.is_dir() else []
        if len(mkvs) != 1:
            skipped.append(action)
            continue
        if str(mkvs[0]) in corrupt:
            # Unfinalized recording (missing moov atom): cannot be decoded,
            # exclude it from the session video like a missing recording.
            skipped.append(f"{action} (corrupt)")
            continue
        sources.append(mkvs[0])

    out_path = out_root / f"episode_{ep_idx:06d}_{stream}_allintra.mp4"
    if not sources:
        return {"episode": ep_idx, "stream": stream, "session": session_name,
                "status": "no_sources", "skipped": skipped}

    if verify_only and out_path.exists():
        pass  # keep the existing video; only verify below
    else:
        list_file = out_root / f".concat_{ep_idx:06d}_{stream}.txt"
        list_file.write_text("".join(f"file '{p}'\n" for p in sources))

        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
               "-i", str(list_file), *ENCODE_ARGS, str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        list_file.unlink(missing_ok=True)
        if proc.returncode != 0:
            return {"episode": ep_idx, "stream": stream, "session": session_name,
                    "status": "failed", "stderr": proc.stderr[-500:]}

    expected = sum(_mkv_frames(p) for p in sources)
    actual = _mkv_frames(out_path)
    ok = actual == expected
    return {"episode": ep_idx, "stream": stream, "session": session_name,
            "status": "ok" if ok else "frame_mismatch",
            "actions": len(sources), "skipped": skipped,
            "expected_frames": expected, "actual_frames": actual,
            "size_bytes": out_path.stat().st_size}


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--showee-root", type=Path, nargs="+", required=True)
    ap.add_argument("--allintra-root", type=Path, required=True)
    ap.add_argument("--streams", nargs="+",
                    default=["wrist_left", "wrist_right", "zed"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--verify-only", action="store_true",
                    help="Skip encoding when the output video exists; only "
                         "re-check frame counts.")
    ap.add_argument("--sessions", type=str, default=None,
                    help="Comma-separated session episode indices (default: all)")
    ap.add_argument("--corrupt-list", type=Path, default=None,
                    help="JSON list of source MKV paths to skip (unfinalized "
                         "recordings that ffprobe cannot open)")
    args = ap.parse_args()

    corrupt: set[str] = set()
    if args.corrupt_list is not None:
        import json as _json
        corrupt = set(_json.loads(args.corrupt_list.read_text()))

    md = np.load(args.memmap_dir / "metadata.npz", allow_pickle=False)
    parquet = _clean(md["episode_source_parquet"])
    tasks = []
    for ep_idx, session in enumerate(parquet):
        if not session or _find_session_dir(args.showee_root, session) is None:
            continue
        if args.sessions and ep_idx not in {int(x) for x in args.sessions.split(",")}:
            continue
        for stream in args.streams:
            tasks.append((ep_idx, session, stream))
    print(f"{len(tasks)} build tasks")

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(build_one, s, st, args.showee_root,
                          args.allintra_root, e, corrupt,
                          args.verify_only): (e, s, st)
                for e, s, st in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            status = r["status"]
            print(f"[{i}/{len(tasks)}] ep{r['episode']} {r['stream']:12s} "
                  f"{r['session']}: {status}", flush=True)
            if status in ("ok", "frame_mismatch"):
                print(f"    actions={r.get('actions')} frames="
                      f"{r.get('actual_frames')}/{r.get('expected_frames')} "
                      f"size={r.get('size_bytes', 0)/1e9:.2f}GB", flush=True)
            elif status == "failed":
                print(f"    {r.get('stderr', '')[-200:]}", flush=True)

    ok = sum(1 for r in results if r["status"] == "ok")
    mismatch = sum(1 for r in results if r["status"] == "frame_mismatch")
    failed = sum(1 for r in results if r["status"] == "failed")
    nosrc = sum(1 for r in results if r["status"] == "no_sources")
    total_gb = sum(r.get("size_bytes", 0) for r in results) / 1e9
    print(f"\nok={ok} frame_mismatch={mismatch} failed={failed} "
          f"no_sources={nosrc} total={total_gb:.1f}GB")
    report = args.allintra_root / "session_videos_report.json"
    with report.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"report: {report}")


if __name__ == "__main__":
    main()
