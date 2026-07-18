#!/usr/bin/env python3
"""Visualize EMG value distributions to judge whether abnormal episodes are outliers.

Renders a 4-panel diagnostic figure covering all dimensions:
  1. Histogram: raw vs filtered_paper, left vs right, normal vs abnormal episodes
  2. Per-channel boxplot: right-hand filtered_paper, normal vs abnormal episodes
  3. Per-episode std bar chart (left vs right, all 41 episodes)
  4. Fraction of |value| > 100 per episode (the anomaly indicator)

Usage:
  python scripts/viz/plot_emg_value_distribution.py
  python scripts/viz/plot_emg_value_distribution.py --out figures/emg_dist.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = "data/EgoEMG_memmap"
N = 66161725

# Abnormal episodes (right-hand |v|>100 fraction > 0.1%), identified in analysis.
ABNORMAL_EPS = {28, 24, 5, 17, 8, 14, 40, 13, 15, 37, 7, 20}
# Heavily abnormal subset (top-4 by severity) for emphasis.
HEAVY_ABN = {28, 24, 5, 17}


def load_episode_index() -> np.ndarray:
    return np.memmap(f"{DATA}/episode_index.dat", dtype=np.int64, mode="r", shape=(N,))


def load_field(hand: str, field: str) -> np.ndarray:
    return np.memmap(f"{DATA}/emg_{hand}_{field}.dat", dtype=np.float32, mode="r", shape=(N, 8))


def episode_bounds(episode_index: np.ndarray) -> dict[int, tuple[int, int]]:
    out = {}
    for ep in np.unique(episode_index):
        idxs = np.where(episode_index == ep)[0]
        out[int(ep)] = (int(idxs.min()), int(idxs.max() + 1))
    return out


def sample_episode(arr: np.ndarray, s: int, e: int, max_n: int = 300_000, seed: int = 0) -> np.ndarray:
    """Random subsample frames from [s, e) for histogram/boxplot (memory-friendly)."""
    n = e - s
    if n <= max_n:
        return arr[s:e].astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_n, replace=False))
    return arr[s + idx].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/emg_value_distribution.png")
    ap.add_argument("--max-samples", type=int, default=300_000,
                    help="Frames to sample per group for histograms/boxplots")
    args = ap.parse_args()

    ei = load_episode_index()
    bounds = episode_bounds(ei)
    print(f"Loaded episode index: {len(bounds)} episodes")

    # Pre-categorize episodes
    normal_eps = [ep for ep in bounds if ep not in ABNORMAL_EPS]
    abn_eps = sorted(ABNORMAL_EPS)

    fig = plt.figure(figsize=(17, 13), dpi=120)
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    # ──────────────────────────────────────────────────────────────────────
    # Panel 1 (top, span 2 cols): Histogram — raw vs filtered, L vs R, normal vs abnormal
    # ──────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])

    def collect(hand, field, eps, max_n):
        arr = load_field(hand, field)
        parts = []
        budget = max_n
        for ep in eps:
            if budget <= 0:
                break
            s, e = bounds[ep]
            take = min(e - s, budget)
            parts.append(sample_episode(arr, s, e, max_n=take, seed=ep))
            budget -= take
        return np.concatenate(parts).ravel() if parts else np.array([])

    mn = args.max_samples
    configs = [
        ("Left · raw",          "left",  "raw",            normal_eps, "#4C72B0", "-",  False),
        ("Left · filtered",     "left",  "filtered_paper", normal_eps, "#4C72B0", "--", False),
        ("Right · raw (normal)","right", "raw",            normal_eps, "#55A868", "-",  False),
        ("Right · filt (normal)","right","filtered_paper", normal_eps, "#55A868", "--", False),
        ("Right · raw (abnormal)","right","raw",           abn_eps,    "#C44E52", "-",  True),
        ("Right · filt (abnormal)","right","filtered_paper",abn_eps,   "#C44E52", "--", True),
    ]
    bins = np.linspace(-150, 150, 151)
    for label, hand, field, eps, color, ls, emphasis in configs:
        vals = collect(hand, field, eps, mn)
        if vals.size == 0:
            continue
        ax1.hist(vals, bins=bins, density=True, histtype="step", linestyle=ls,
                 color=color, linewidth=2.4 if emphasis else 1.6,
                 label=f"{label} (std={vals.std():.1f})", alpha=0.95 if emphasis else 0.8)
    ax1.axvline(100, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.axvline(-100, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.set_xlabel("EMG value", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Value distribution: L/R × raw/filtered × normal/abnormal episodes",
                  fontsize=12.5, weight="bold")
    ax1.set_yscale("log")
    ax1.set_xlim(-150, 150)
    ax1.legend(fontsize=8.5, loc="upper right", ncol=1, framealpha=0.95)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    # ──────────────────────────────────────────────────────────────────────
    # Panel 2 (top-right): per-channel boxplot, right-hand filtered_paper
    # ──────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    arr_r_fp = load_field("right", "filtered_paper")
    normal_vals = collect("right", "filtered_paper", normal_eps, mn)  # already pooled, but redo per-channel
    # per-channel: sample frames and keep channel dim
    def per_channel(eps, max_n):
        parts = []
        budget = max_n
        for ep in eps:
            if budget <= 0: break
            s, e = bounds[ep]
            take = min(e - s, budget)
            parts.append(sample_episode(arr_r_fp, s, e, max_n=take, seed=ep))
            budget -= take
        return np.concatenate(parts) if parts else np.empty((0, 8))

    nv = per_channel(normal_eps, mn)
    av = per_channel(abn_eps, mn)
    box_data, positions, colors = [], [], []
    for ch in range(8):
        box_data.append(nv[:, ch]); positions.append(ch * 3 + 0); colors.append("#55A868")
        box_data.append(av[:, ch]); positions.append(ch * 3 + 1.2); colors.append("#C44E52")
    bp = ax2.boxplot(box_data, positions=positions, widths=1.0, showfliers=False,
                     patch_artist=True, manage_ticks=False,
                     medianprops=dict(color="black", linewidth=1.3))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.set_xticks([ch * 3 + 0.6 for ch in range(8)])
    ax2.set_xticklabels([f"ch{i}" for i in range(8)])
    ax2.set_ylabel("Right filtered_paper value")
    ax2.set_title("Per-channel: normal vs abnormal", fontsize=11, weight="bold")
    ax2.axhline(100, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax2.axhline(-100, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color="#55A868", alpha=0.7, label="Normal eps"),
                        Patch(color="#C44E52", alpha=0.7, label="Abnormal eps")],
               fontsize=9, loc="upper right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    # ──────────────────────────────────────────────────────────────────────
    # Panel 3 (middle, full width): per-episode std, left vs right (filtered_paper)
    # ──────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    arr_l_fp = load_field("left", "filtered_paper")
    eps_sorted = sorted(bounds.keys())
    l_stds, r_stds, ep_colors = [], [], []
    for ep in eps_sorted:
        s, e = bounds[ep]
        # stream std (avoid loading full episode)
        chunk = 1_000_000
        l_sq = 0.0; l_n = 0
        for i in range(s, e, chunk):
            j = min(i + chunk, e)
            b = arr_l_fp[i:j].astype(np.float64)
            l_sq += (b ** 2).sum(); l_n += b.size
        l_stds.append(np.sqrt(l_sq / l_n))
        r_sq = 0.0; r_n = 0
        for i in range(s, e, chunk):
            j = min(i + chunk, e)
            b = arr_r_fp[i:j].astype(np.float64)
            r_sq += (b ** 2).sum(); r_n += b.size
        r_stds.append(np.sqrt(r_sq / r_n))
        ep_colors.append("#C44E52" if ep in ABNORMAL_EPS else "#888888")

    x = np.arange(len(eps_sorted))
    w = 0.4
    ax3.bar(x - w/2, l_stds, width=w, color="#4C72B0", alpha=0.85, label="Left filtered")
    # color right bars by abnormal/normal
    r_normal = [r if ep not in ABNORMAL_EPS else 0 for ep, r in zip(eps_sorted, r_stds)]
    r_abn = [r if ep in ABNORMAL_EPS else 0 for ep, r in zip(eps_sorted, r_stds)]
    ax3.bar(x + w/2, r_normal, width=w, color="#55A868", alpha=0.85, label="Right filtered (normal ep)")
    ax3.bar(x + w/2, r_abn, width=w, color="#C44E52", alpha=0.9, label="Right filtered (abnormal ep)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(ep) for ep in eps_sorted], fontsize=9)
    ax3.set_xlabel("Episode index", fontsize=11)
    ax3.set_ylabel("Std (filtered_paper)", fontsize=11)
    ax3.set_title("Per-episode std: left vs right (abnormal right episodes in red)",
                  fontsize=12.5, weight="bold")
    ax3.legend(fontsize=9.5, loc="upper right", ncol=3)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.3)
    # mark heavy abnormal episodes
    for ep in HEAVY_ABN:
        xi = eps_sorted.index(ep)
        ax3.annotate(f"ep{ep}", (xi + w/2, r_stds[xi]), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8.5, color="#C44E52", weight="bold")

    # ──────────────────────────────────────────────────────────────────────
    # Panel 4 (bottom, full width): fraction of |v|>100 per episode, left vs right
    # ──────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    l_big, r_big = [], []
    for ep in eps_sorted:
        s, e = bounds[ep]
        chunk = 1_000_000
        l_cnt = l_tot = r_cnt = r_tot = 0
        for i in range(s, e, chunk):
            j = min(i + chunk, e)
            bl = arr_l_fp[i:j]; br = arr_r_fp[i:j]
            l_cnt += (np.abs(bl) > 100).sum(); l_tot += bl.size
            r_cnt += (np.abs(br) > 100).sum(); r_tot += br.size
        l_big.append(l_cnt / max(l_tot, 1) * 100)
        r_big.append(r_cnt / max(r_tot, 1) * 100)

    ax4.bar(x - w/2, l_big, width=w, color="#4C72B0", alpha=0.85, label="Left")
    rn = [r if ep not in ABNORMAL_EPS else 0 for ep, r in zip(eps_sorted, r_big)]
    ra = [r if ep in ABNORMAL_EPS else 0 for ep, r in zip(eps_sorted, r_big)]
    ax4.bar(x + w/2, rn, width=w, color="#55A868", alpha=0.85, label="Right (normal ep)")
    ax4.bar(x + w/2, ra, width=w, color="#C44E52", alpha=0.9, label="Right (abnormal ep)")
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(ep) for ep in eps_sorted], fontsize=9)
    ax4.set_xlabel("Episode index", fontsize=11)
    ax4.set_ylabel("% of samples with |value| > 100", fontsize=11)
    ax4.set_title("Anomaly indicator: fraction of large-magnitude samples per episode",
                  fontsize=12.5, weight="bold")
    ax4.set_yscale("log")
    ax4.set_ylim(1e-4, max(r_big) * 1.5)
    ax4.legend(fontsize=9.5, loc="upper left", ncol=3)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.3)
    for ep in HEAVY_ABN:
        xi = eps_sorted.index(ep)
        ax4.annotate(f"{r_big[xi]:.1f}%", (xi + w/2, r_big[xi]),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=8, color="#C44E52", weight="bold")

    fig.suptitle("EMG Value Distribution Diagnosis — EgoEMG filtered_paper (66M frames, 41 episodes)",
                 fontsize=14.5, weight="bold", y=0.995)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
