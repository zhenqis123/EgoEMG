#!/usr/bin/env python
#
# Aggregate per-file EMG stats into user/stage/side/channel summaries.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-file EMG stats.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/emg_stats_per_file.csv"),
        help="Input CSV produced by emg_stats_per_file.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/emg_stats_aggregate.csv"),
        help="Output aggregated CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if df.empty:
        print("Input is empty; nothing to aggregate.")
        return

    group_cols = ["user", "stage", "side", "channel"]
    num_cols = [
        c
        for c in df.columns
        if c
        not in {
            "file",
            "user",
            "stage",
            "side",
            "channel",
        }
    ]

    agg_df = df.groupby(group_cols)[num_cols].agg(
        ["mean", "median", "std", "min", "max"]
    )
    agg_df.columns = ["_".join(col).rstrip("_") for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(args.output, index=False)
    print(f"Wrote aggregated stats to {args.output} (rows={len(agg_df)})")


if __name__ == "__main__":
    main()
