#!/usr/bin/env python
#
# Visualize EMG stats (heatmaps/boxplots) from aggregated CSV.

from __future__ import annotations

import argparse
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")  # safe for headless by default
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize aggregated EMG stats.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/emg_stats_aggregate.csv"),
        help="Aggregated CSV (from emg_stats_aggregate.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/emg_stats_plots"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (headless-safe default is to only save).",
    )
    return parser.parse_args()


def heatmap_user_channel(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    if "rms_mean" not in df.columns:
        return
    pivot = df.pivot_table(
        index="user", columns="channel", values="rms_mean", aggfunc="mean"
    )
    plt.figure(figsize=(10, max(4, len(pivot) * 0.2)))
    sns.heatmap(pivot, cmap="mako", cbar_kws={"label": "RMS (mean)"})
    plt.title("User × Channel RMS (mean of means)")
    plt.tight_layout()
    path = out_dir / "heatmap_user_channel_rms.png"
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def heatmap_stage_channel(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    if "rms_mean" not in df.columns:
        return
    pivot = df.pivot_table(
        index="stage", columns="channel", values="rms_mean", aggfunc="mean"
    )
    plt.figure(figsize=(10, max(4, len(pivot) * 0.2)))
    sns.heatmap(pivot, cmap="mako", cbar_kws={"label": "RMS (mean)"})
    plt.title("Stage × Channel RMS (mean of means)")
    plt.tight_layout()
    path = out_dir / "heatmap_stage_channel_rms.png"
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def boxplot_channel(df: pd.DataFrame, out_dir: Path, metric: str, hue: str | None = None, show: bool = False) -> None:
    if metric not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="channel", y=metric, hue=hue)
    plt.title(f"{metric} by channel" + (f" (hue={hue})" if hue else ""))
    plt.tight_layout()
    fname = f"boxplot_{metric}_by_channel" + (f"_by_{hue}" if hue else "") + ".png"
    path = out_dir / fname
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def corr_heatmap(df: pd.DataFrame, out_dir: Path, group_col: str, metric: str, show: bool) -> None:
    """Plot channel correlation heatmap averaged over a grouping."""
    if metric not in df.columns or group_col not in df.columns:
        return
    pivot = df.pivot_table(index=group_col, columns="channel", values=metric, aggfunc="mean")
    if pivot.shape[0] == 0:
        return
    corr = pivot.transpose().corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, cmap="mako", vmin=-1, vmax=1, center=0, cbar_kws={"label": f"corr({metric})"})
    plt.title(f"Channel correlation of {metric} aggregated by {group_col}")
    plt.tight_layout()
    fname = f"corr_heatmap_{metric}_by_{group_col}.png"
    path = out_dir / fname
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def distribution_hist(df: pd.DataFrame, out_dir: Path, metric: str, hue: str | None, show: bool) -> None:
    if metric not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=metric, hue=hue, element="step", stat="density", common_norm=False, bins=50)
    plt.title(f"Distribution of {metric}" + (f" (hue={hue})" if hue else ""))
    plt.tight_layout()
    fname = f"hist_{metric}" + (f"_by_{hue}" if hue else "") + ".png"
    path = out_dir / fname
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def pca_scatter(df: pd.DataFrame, out_dir: Path, value_cols: list[str], color_col: str, show: bool) -> None:
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return
    if not all(c in df.columns for c in value_cols) or color_col not in df.columns:
        return
    X = df[value_cols].to_numpy(dtype=float)
    if X.shape[0] < 2:
        return
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    plot_df = pd.DataFrame({"pc1": comps[:, 0], "pc2": comps[:, 1], color_col: df[color_col]})
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue=color_col, s=20, alpha=0.7)
    plt.title(f"PCA of channel stats colored by {color_col}")
    plt.tight_layout()
    fname = f"pca_scatter_by_{color_col}.png"
    path = out_dir / fname
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if df.empty:
        print("Input is empty; nothing to visualize.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_user_channel(df, args.output_dir, args.show)
    heatmap_stage_channel(df, args.output_dir, args.show)
    for metric in ("rms_mean", "energy_mean", "std_mean"):
        boxplot_channel(df, args.output_dir, metric, show=args.show)
        boxplot_channel(df, args.output_dir, metric, hue="side", show=args.show)
        distribution_hist(df, args.output_dir, metric, hue="side", show=args.show)
        distribution_hist(df, args.output_dir, metric, hue="stage", show=args.show)

    # Correlation heatmaps (channel-wise, aggregated by user/stage)
    for group_col in ("user", "stage"):
        for metric in ("rms_mean", "energy_mean"):
            corr_heatmap(df, args.output_dir, group_col, metric, show=args.show)

    # PCA scatter using a few metrics as features
    feature_cols = [c for c in df.columns if c.endswith("_mean") and any(k in c for k in ("rms", "energy", "std"))]
    if feature_cols:
        pca_scatter(df, args.output_dir, feature_cols, color_col="user", show=args.show)
        pca_scatter(df, args.output_dir, feature_cols, color_col="stage", show=args.show)

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
