#!/usr/bin/env python3
"""Compute field-aware per-hand EMG normalization statistics for ShowEE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads((args.memmap_root / "manifest.json").read_text())
    total = int(manifest["total_rows"])
    count = min(total, args.num_samples)
    rows = np.sort(np.random.default_rng(args.seed).choice(total, count, replace=False))
    hand_stats = {}
    combined_sum = combined_sum2 = 0.0
    combined_count = 0
    for hand in ("left", "right"):
        name = f"emg_{hand}_filtered_paper"
        info = manifest["fields"][name]
        data = np.memmap(
            args.memmap_root / info["filename"], mode="r", dtype=info["dtype"], shape=tuple(info["shape"])
        )
        sum_x = np.zeros(8, dtype=np.float64)
        sum_x2 = np.zeros(8, dtype=np.float64)
        for offset in range(0, count, 100_000):
            block = np.asarray(data[rows[offset : offset + 100_000]], dtype=np.float64)
            sum_x += block.sum(axis=0)
            sum_x2 += np.square(block).sum(axis=0)
        mean = sum_x / count
        variance = np.maximum(sum_x2 / count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        scalar_mean = float(sum_x.sum() / (count * 8))
        scalar_variance = float(sum_x2.sum() / (count * 8) - scalar_mean**2)
        hand_stats[hand] = {
            "mean": scalar_mean,
            "std": float(np.sqrt(max(scalar_variance, 0.0))),
            "per_channel_mean": mean.tolist(),
            "per_channel_std": std.tolist(),
            "num_sampled_rows": count,
        }
        combined_sum += float(sum_x.sum())
        combined_sum2 += float(sum_x2.sum())
        combined_count += count * 8
    mean = combined_sum / combined_count
    std = np.sqrt(max(combined_sum2 / combined_count - mean**2, 0.0))
    result = {
        "showee__filtered_paper": {"mean": mean, "std": std},
        "showee__filtered_paper_left": hand_stats["left"],
        "showee__filtered_paper_right": hand_stats["right"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
