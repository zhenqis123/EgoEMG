"""Detect webcam recording freezes per episode using decord frame-difference analysis.

For each episode, computes diff(frame[i], frame[i+2]) across the video.
During a freeze, diffs are near zero. When the freeze ends at frame k,
diff(k-2, k) spikes — much larger than typical motion diffs.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

MEMMAP_DIR = "/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap"
VIDEO_ROOT = "/mnt/nvme/xiziheng/training_dataset_lerobot_full_NEW"


def load_metadata():
    md = dict(np.load(Path(MEMMAP_DIR) / "metadata.npz", allow_pickle=False))
    episode_ids = [
        x.decode("utf-8").rstrip("\x00") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in md["episode_id"]
    ]
    video_paths = [
        x.decode("utf-8").rstrip("\x00") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in md["episode_webcam_video_path"]
    ]
    return episode_ids, video_paths


def analyze_episode(video_path: str, gap: int = 2,
                    freeze_diff_max: float = 1.5,
                    resume_diff_min: float = 6.0,
                    min_freeze_len: int = 3,
                    downsample: int = 4):
    """Analyze a single episode video for recording freezes.

    Uses decord sequential iterator (fastest decode path) and optional
    spatial downsampling for faster diff computation.

    Args:
        video_path: path to webcam mp4
        gap: frame gap for difference computation (default 2)
        freeze_diff_max: max mean pixel diff to consider a frame pair "frozen"
        resume_diff_min: min diff to consider "freeze ended" (spike)
        min_freeze_len: minimum number of consecutive frozen frames to report
        downsample: spatial downsample factor (default 4 → 1/16 pixels)

    Returns:
        dict with total_frames, freeze_segments [(start, end), ...], num_frozen
    """
    import decord

    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total < gap + 1:
        return {"total_frames": total, "freeze_segments": [], "num_frozen": 0}

    # Sequential frame iteration — decord's fastest path (no seeking)
    all_diffs = []
    prev2_frame = None
    prev1_frame = None

    for i in tqdm(range(total), desc="  scanning frames", unit="f",
                   leave=False, ncols=80):
        cur = vr[i].asnumpy().astype(np.float32)
        # Spatial downsample for faster diff (skip=downsample in both axes)
        if downsample > 1:
            cur = cur[::downsample, ::downsample]

        if prev2_frame is not None:
            diff = np.abs(prev2_frame - cur).mean()
            all_diffs.append((i, diff))
        prev2_frame = prev1_frame
        prev1_frame = cur

    diffs = np.array([d for _, d in all_diffs], dtype=np.float64)
    indices = np.array([i for i, _ in all_diffs], dtype=np.int64)

    if len(diffs) == 0:
        return {"total_frames": total, "freeze_segments": [], "num_frozen": 0}

    # Find "especially large" diffs → freeze-end events
    median_diff = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median_diff)))
    # Use a robust threshold: median + 10 * MAD, but floor at resume_diff_min
    spike_threshold = max(median_diff + 10 * mad, resume_diff_min)

    # Rely on the gap-diffs: a freeze segment is a run where gap-diffs < freeze_diff_max,
    # bounded by frames where gap-diff > spike_threshold (resume event)
    frozen_mask = diffs < freeze_diff_max

    # Find contiguous frozen segments
    segments = []
    seg_start = None
    for i, is_frozen in enumerate(frozen_mask):
        frame_idx = indices[i]
        if is_frozen:
            if seg_start is None:
                seg_start = frame_idx
        else:
            if seg_start is not None:
                seg_end = frame_idx  # frame_idx is the first non-frozen frame
                if seg_end - seg_start >= min_freeze_len:
                    segments.append((int(seg_start), int(seg_end)))
                seg_start = None
    if seg_start is not None:
        # Freeze extends to end of video
        seg_end = total
        if seg_end - seg_start >= min_freeze_len:
            segments.append((int(seg_start), int(seg_end)))

    # Merge adjacent segments that are separated by a single spike frame
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    num_frozen = sum(end - start for start, end in merged)
    return {
        "total_frames": total,
        "freeze_segments": merged,
        "num_frozen": num_frozen,
        "median_diff": float(median_diff),
        "spike_threshold": float(spike_threshold),
        "max_diff": float(np.max(diffs)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect webcam video freeze segments per episode"
    )
    parser.add_argument("--episodes", type=str, nargs="*", default=None,
                        help="Episode IDs to analyze (default: all)")
    parser.add_argument("--gap", type=int, default=2,
                        help="Frame gap for diff computation")
    parser.add_argument("--freeze-diff-max", type=float, default=1.5,
                        help="Max mean pixel diff to treat as frozen")
    parser.add_argument("--resume-diff-min", type=float, default=6.0,
                        help="Min diff to treat as freeze-end spike")
    parser.add_argument("--min-freeze-len", type=int, default=3,
                        help="Min consecutive frozen frames to report")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Spatial downsample factor (default 4 → 1/16 pixels)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to path")
    args = parser.parse_args()

    episode_ids, video_paths = load_metadata()

    if args.episodes:
        selected = [(i, eid, vp) for i, eid, vp in
                    zip(range(len(episode_ids)), episode_ids, video_paths)
                    if eid in args.episodes]
    else:
        selected = [(i, eid, vp) for i, eid, vp in
                    zip(range(len(episode_ids)), episode_ids, video_paths)]

    report = {}
    for idx, ep_id, video_rel in tqdm(selected, desc="episodes", unit="ep"):
        video_path = Path(VIDEO_ROOT) / video_rel
        if not video_path.exists():
            print(f"[{ep_id}] SKIP: video not found: {video_path}")
            continue

        result = analyze_episode(
            str(video_path),
            gap=args.gap,
            freeze_diff_max=args.freeze_diff_max,
            resume_diff_min=args.resume_diff_min,
            min_freeze_len=args.min_freeze_len,
            downsample=args.downsample,
        )

        report[ep_id] = result

        # Print summary
        total = result["total_frames"]
        num_frozen = result["num_frozen"]
        pct = 100 * num_frozen / total if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"[{ep_id}]  total={total}, frozen={num_frozen} ({pct:.1f}%)")
        if "median_diff" in result:
            print(f"  median gap-{args.gap} diff = {result['median_diff']:.2f}, "
                  f"spike threshold = {result['spike_threshold']:.2f}, "
                  f"max diff = {result['max_diff']:.2f}")
        if result["freeze_segments"]:
            print(f"  freeze segments ({len(result['freeze_segments'])}):")
            for start, end in result["freeze_segments"]:
                duration_s = (end - start) / 30.0  # assuming 30fps
                print(f"    frames [{start:>6d} – {end:>6d}]  "
                      f"len={end-start:>4d}  ({duration_s:.1f}s)")
        else:
            print(f"  no freeze segments detected")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
