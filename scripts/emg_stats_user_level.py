#!/usr/bin/env python
#
# Aggregate per-channel stats into per-user summaries and visualize cross-user distributions.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe by default
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate channel stats to user-level and visualize cross-user distributions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/emg_stats_aggregate.csv"),
        help="Aggregated CSV from emg_stats_aggregate.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/emg_stats_user_level"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (default: only save).",
    )
    return parser.parse_args()


def aggregate_user(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.columns if c not in {"user", "stage", "side", "channel", "file"}]
    agg = df.groupby("user")[num_cols].agg(["mean", "median", "std", "min", "max"])
    agg.columns = ["_".join(col).rstrip("_") for col in agg.columns.values]
    agg.reset_index(inplace=True)
    return agg


def boxplot_users(df_user: pd.DataFrame, metric: str, out_dir: Path, show: bool) -> None:
    if metric not in df_user.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_user, y=metric)
    plt.title(f"User-level distribution of {metric}")
    plt.tight_layout()
    path = out_dir / f"user_boxplot_{metric}.png"
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def hist_users(df_user: pd.DataFrame, metric: str, out_dir: Path, show: bool) -> None:
    if metric not in df_user.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_user, x=metric, bins=50, kde=True)
    plt.title(f"User-level histogram of {metric}")
    plt.tight_layout()
    path = out_dir / f"user_hist_{metric}.png"
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if df.empty:
        print("Input is empty; nothing to process.")
        return

    df_user = aggregate_user(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Choose representative metrics at user level
    metrics = [
        "rms_mean_mean",
        "energy_mean_mean",
        "std_mean_mean",
        "finite_frac_mean",
    ]

    for metric in metrics:
        hist_users(df_user, metric, args.output_dir, args.show)
        boxplot_users(df_user, metric, args.output_dir, args.show)

    df_user.to_csv(args.output_dir / "user_level_stats.csv", index=False)
    print(f"Wrote user-level stats and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
