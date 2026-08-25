"""Finalize the release metadata for the wrist/ZED ShowEE streams.

Adds the episode-level video path keys for the newly built streams and
fills the ShowEE session entries::

    episode_wrist_left_video_path   episode_XXXXXX_wrist_left.mp4
    episode_wrist_right_video_path  episode_XXXXXX_wrist_right.mp4
    episode_zed_video_path          episode_XXXXXX_zed.mp4

for the 22 ShowEE sessions (episodes 41..62).  EgoEMG / Incre episodes
keep empty paths (no wrist views; their ZED videos are not wired into
the memmap).  Run after ``build_showee_session_videos.py`` and
``build_showee_wrist_zed_indices.py``.

Usage::

    python scripts/prepare/finalize_release_metadata.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_full_memmap
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    path = args.memmap_dir / "metadata.npz"
    md = np.load(path, allow_pickle=False)
    parquet = np.asarray(
        [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
         for v in md["episode_source_parquet"]])
    # ShowEE sessions: parquet == session dir name (no slash).
    showee = np.asarray(
        [bool(p) and "/" not in p for p in parquet])

    out = {k: v for k, v in md.items()}
    for stream in ("wrist_left", "wrist_right", "zed"):
        key = f"episode_{stream}_video_path"
        existing = out.get(key)
        values = np.asarray(
            [f"episode_{i:06d}_{stream}.mp4" if showee[i]
             else (existing[i] if existing is not None else "")
             for i in range(len(parquet))], dtype=object)
        values = np.asarray([v.encode() if isinstance(v, str) else v
                             for v in values])
        out[key] = values
        n = int(showee.sum())
        print(f"{key}: {n} ShowEE entries filled")

    if args.dry_run:
        print("DRY RUN: no changes written")
        return
    backup = path.with_suffix(".npz.bak3")
    if not backup.exists():
        shutil.copy2(path, backup)
    np.savez(path, **out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
