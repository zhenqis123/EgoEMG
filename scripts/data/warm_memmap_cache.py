#!/usr/bin/env python
"""Sequentially read selected splits of a memmap corpus into the page cache.

Random window access into a multi-hundred-GB memmap is disk-bound on first
touch. Warming the page cache with one large sequential read per session
range beforehand lets DataLoader workers serve training reads at RAM speed
from epoch 1. Page cache is reclaimable, so this never risks OOM.

Example:
    python scripts/data/warm_memmap_cache.py \
        --memmap-dir data/emg2pose_memmap --splits train
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    args = parser.parse_args()

    with open(args.memmap_dir / "manifest.json") as f:
        manifest = json.load(f)
    meta = np.load(args.memmap_dir / "metadata.npz")

    split_names = [s.decode() if isinstance(s, bytes) else str(s) for s in meta["splits_split"]]
    wanted = {split_names.index(s) for s in args.splits if s in split_names}
    mask = np.isin(meta["session_split_id"], list(wanted)) if wanted else np.zeros(len(meta["session_start_idx"]), bool)
    ranges = sorted(
        zip(meta["session_start_idx"][mask].tolist(), meta["session_end_idx"][mask].tolist())
    )
    rows = sum(e - s for s, e in ranges)
    print(f"Warming {len(ranges)} sessions ({rows:,} rows) of splits {args.splits}")

    chunk = np.empty(64 << 20, dtype=np.uint8)  # 64 MB read buffer
    t0 = time.time()
    total = 0
    for field, info in manifest["fields"].items():
        shape = info["shape"]
        row_bytes = int(np.dtype(info["dtype"]).itemsize) * (
            int(shape[1]) if len(shape) > 1 else 1
        )
        path = args.memmap_dir / info["filename"]
        with open(path, "rb", buffering=0) as f:
            for start, end in ranges:
                pos = start * row_bytes
                remaining = (end - start) * row_bytes
                f.seek(pos)
                while remaining > 0:
                    n = f.readinto(chunk[: min(len(chunk), remaining)])
                    if not n:
                        raise EOFError(f"short read on {path} at row {start}")
                    remaining -= n
                    total += n
        dt = time.time() - t0
        print(f"  {info['filename']}: cumulative {total / 1e9:.1f} GB in {dt:.0f}s ({total / dt / 1e9:.2f} GB/s)")
    print(f"Done: {total / 1e9:.1f} GB warmed in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
