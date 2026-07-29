#!/usr/bin/env python3
"""Compute per-dataset mean/std for emg2pose 8ch subset.

Reads the emg2pose memmap EMG, slices channels via channel_indices,
and estimates mean/std from a random subset. Results are printed
in JSON format ready for per_dataset_norm_stats.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHANNEL_INDICES = [10, 12, 0, 1, 2, 4, 5, 6]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute emg2pose 8ch norm stats from memmap"
    )
    p.add_argument(
        "--memmap-dir",
        default="./data/emg_corpus/emg2pose_v3_memmap",
    )
    p.add_argument("--num-samples", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    memmap_dir = Path(args.memmap_dir)
    emg_path = memmap_dir / "emg.dat"

    with open(memmap_dir / "manifest.json") as f:
        manifest = json.load(f)

    emg_info = manifest["fields"]["emg"]
    n_total = emg_info["shape"][0]
    n_channels = emg_info["shape"][1]
    dtype = np.dtype(emg_info["dtype"])

    print(f"EMG: {n_total:,} samples × {n_channels} channels", file=sys.stderr)

    indices = np.asarray(CHANNEL_INDICES, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= n_channels):
        raise ValueError(f"channel_indices out of range for {n_channels}-channel data: {CHANNEL_INDICES}")

    # Sample random rows
    rng = np.random.default_rng(args.seed)
    n_sample = min(args.num_samples, n_total)
    sample_rows = np.sort(rng.choice(n_total, size=n_sample, replace=False))

    emg_mm = np.memmap(str(emg_path), dtype=dtype, mode="r", shape=(n_total, n_channels))

    # Read in blocks to avoid memory pressure
    block_size = 100_000
    sum_x = np.zeros(len(indices), dtype=np.float64)
    sum_x2 = np.zeros(len(indices), dtype=np.float64)
    count = 0

    for i in range(0, n_sample, block_size):
        block_rows = sample_rows[i : i + block_size]
        block = emg_mm[block_rows, :].astype(np.float64)  # (B, 16)
        block_8ch = block[:, indices]  # (B, 8)
        sum_x += block_8ch.sum(axis=0)
        sum_x2 += (block_8ch ** 2).sum(axis=0)
        count += len(block_rows)
        if (i // block_size) % 20 == 0:
            print(f"  {count:,} / {n_sample:,} ...", file=sys.stderr)

    mean = sum_x / count
    variance = (sum_x2 / count) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0))

    # Aggregate to scalars
    scalar_mean = float(mean.mean())
    scalar_std = float(np.sqrt((variance.sum() + (mean - scalar_mean).sum() ** 2) / len(indices)))

    print(f"\nDone. {count:,} samples.", file=sys.stderr)
    print(f"  Per-channel mean: {mean}", file=sys.stderr)
    print(f"  Per-channel std:  {std}", file=sys.stderr)
    print(f"  Scalar mean: {scalar_mean:.6f}", file=sys.stderr)
    print(f"  Scalar std:  {scalar_std:.6f}", file=sys.stderr)

    result = {
        "emg2pose_8ch_aligned": {
            "mean": scalar_mean,
            "std": scalar_std,
        }
    }

    if args.output_json:
        # Merge with existing if present
        output_path = Path(args.output_json)
        if output_path.exists():
            existing = json.loads(output_path.read_text())
        else:
            existing = {}
        existing.update(result)
        output_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
        print(f"\nUpdated: {output_path}", file=sys.stderr)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
