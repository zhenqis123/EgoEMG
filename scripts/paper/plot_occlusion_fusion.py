"""Plot occlusion vs fusion gain — publication-quality figure.

Two-panel figure:
  (a) Vision MAE & Fusion MAE vs occlusion, SEM bands, gap shading
  (b) Per-sample delta scatter (hexbin density) + LOESS trend

Usage:
    python scripts/paper/plot_occlusion_fusion.py \
        --csv test_results/occlusion_fusion_results.csv \
        --output test_results/occlusion_fusion_gain.png
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import sem, spearmanr

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-friendly (Wong 2011)
C_VISION = "#D55E00"
C_FUSION = "#009E73"
C_DELTA = "#6A3D9A"
C_GAP = "#E6A0C4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test_results/occlusion_fusion_results.csv")
    parser.add_argument("--output", default="test_results/occlusion_fusion_gain.png")
    parser.add_argument("--min-samples-per-bin", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    valid = df[df["occlusion_score"].notna() & (df["mae_vision_deg"] > 0.01)].copy()

    occ = valid["occlusion_score"].values
    vis = valid["mae_vision_deg"].values
    fus = valid["mae_fusion_deg"].values
    delta = valid["delta_fusion_deg"].values

    # ── Fixed-width bins on x-axis ─────────────────────────────────────────
    bin_width = 0.05
    bin_start = np.floor(occ.min() / bin_width) * bin_width
    bin_end = np.ceil(occ.max() / bin_width) * bin_width
    bin_edges = np.arange(bin_start, bin_end + bin_width / 2, bin_width)

    # Merge sparse bins
    bin_centers = []
    bin_vis_m, bin_vis_s = [], []
    bin_fus_m, bin_fus_s = [], []
    bin_delta_m, bin_delta_s = [], []
    bin_counts = []
    acc = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        idx = np.where((occ >= lo) & (occ < hi))[0]
        acc.append(idx)
        if sum(len(a) for a in acc) < args.min_samples_per_bin and i < len(bin_edges) - 2:
            continue
        merged = np.concatenate(acc)
        if len(merged) < 10:
            acc = []
            continue
        bin_centers.append(float(occ[merged].mean()))
        bin_vis_m.append(vis[merged].mean())
        bin_vis_s.append(sem(vis[merged]))
        bin_fus_m.append(fus[merged].mean())
        bin_fus_s.append(sem(fus[merged]))
        bin_delta_m.append(delta[merged].mean())
        bin_delta_s.append(sem(delta[merged]))
        bin_counts.append(len(merged))
        acc = []

    bc = np.array(bin_centers)
    bv = np.array(bin_vis_m)
    bf = np.array(bin_fus_m)
    bvs = np.array(bin_vis_s)
    bfs = np.array(bin_fus_s)
    cnt = np.array(bin_counts)

    # ── LOESS for delta trend (on binned data, weighted by count) ───────────
    order = np.argsort(bc)
    xs, ys, ws = bc[order], np.array(bin_delta_m)[order], cnt[order]
    spl = UnivariateSpline(xs, ys, w=ws, s=len(xs) * 0.4, k=3)
    x_smooth = np.linspace(bc.min(), bc.max(), 200)
    y_smooth = spl(x_smooth)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.3))

    # ═══════════════ Panel (a): Dual-line MAE vs occlusion ═══════════════════
    # Gap shading
    ax_a.fill_between(bc, bf - bfs, bf + bfs, alpha=0.15, color=C_FUSION)
    ax_a.fill_between(bc, bv - bvs, bv + bvs, alpha=0.15, color=C_VISION)
    ax_a.fill_between(bc, bf, bv, alpha=0.08, color=C_GAP, zorder=1)

    ax_a.plot(bc, bv, "o-", color=C_VISION, linewidth=2.2, markersize=7,
              label="Vision-only", zorder=4)
    ax_a.plot(bc, bf, "s-", color=C_FUSION, linewidth=2.2, markersize=7,
              label="Fusion (EMG+Vision)", zorder=4)

    # Annotate gap at first and last bins
    for sel_idx, x_shift in [(0, 0.06), (len(bc) - 1, 0)]:
        gap = bv[sel_idx] - bf[sel_idx]
        mid_y = (bv[sel_idx] + bf[sel_idx]) / 2
        ax_a.annotate(
            f"$\\Delta$={gap:.2f}°", (bc[sel_idx] + x_shift, mid_y),
            fontsize=14, fontweight="bold", color="#333333", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85),
        )

    ax_a.set_xlabel("Self-Occlusion Score  (0 = fully visible)")
    ax_a.set_ylabel("Mean Joint Angle MAE (°)")
    ax_a.set_title("(a)  Prediction Error vs. Occlusion", fontweight="bold", loc="left")
    ax_a.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#dddddd")
    ax_a.set_xlim(bc.min() - 0.02, bc.max() + 0.02)
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax_a.grid(axis="y", alpha=0.25, linewidth=0.6)

    # ═══════════════ Panel (b): Delta scatter + trend ════════════════════════
    # Light scatter — clean background, no dark hexbin
    n_scatter = min(4000, len(occ))
    rng = np.random.default_rng(42)
    s_idx = rng.choice(len(occ), size=n_scatter, replace=False)
    ax_b.scatter(occ[s_idx], delta[s_idx],
                 s=2.5, color="#B8B8B8", alpha=0.10,
                 edgecolors="none", rasterized=True, zorder=1)

    # Trend line
    ax_b.plot(x_smooth, y_smooth, color=C_DELTA, linewidth=2.8, zorder=4,
              label="LOESS trend")

    # Zero reference
    ax_b.axhline(y=0, color="#777777", linewidth=0.9, linestyle="--",
                 dashes=(6, 4), zorder=2)

    # Spearman annotation
    rho, pval = spearmanr(occ, delta)
    # Avoid reporting a rounded p-value of 0.000.
    p_str = "$p < 0.001$" if pval < 1e-3 else f"$p$ = {pval:.3f}"
    ax_b.text(0.97, 0.07,
              f"Spearman $\\rho = {rho:.3f}$\n{p_str}",
              transform=ax_b.transAxes, ha="right", va="bottom",
              fontsize=10,
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.85))

    # Per-bin markers (show bin confidence)
    ax_b.scatter(bc, np.array(bin_delta_m), s=cnt**0.45, c=C_DELTA,
                 alpha=0.6, edgecolors="white", linewidths=0.5, zorder=3)

    ax_b.set_xlabel("Self-Occlusion Score  (0 = fully visible)")
    ax_b.set_ylabel("Fusion Gain  $\\Delta$ = MAE$_{vision}$ $-$ MAE$_{fusion}$ (°)")
    ax_b.set_title("(b)  Fusion Gain per Sample vs. Occlusion", fontweight="bold", loc="left")
    ax_b.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#dddddd")
    y_lo = max(np.array(bin_delta_m).min() - 0.08, -0.05)
    y_hi = np.array(bin_delta_m).max() + 0.08
    ax_b.set_xlim(bc.min() - 0.02, bc.max() + 0.02)
    ax_b.set_ylim(y_lo, y_hi)
    ax_b.grid(alpha=0.25, linewidth=0.6)

    # ═══════════════ Finalise ══════════════════════════════════════════════════
    plt.tight_layout(pad=1.2)
    fig.savefig(args.output, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved to {args.output}")

    # Print per-bin table
    print(f"\n{'Bin':<10} {'N':>6} {'Vis(°)':>8} {'Fus(°)':>8} {'Δ(°)':>8} {'SEM_Δ':>8}")
    print("-" * 52)
    for i in range(len(bc)):
        print(f"{bc[i]:<10.3f} {cnt[i]:>6} "
              f"{bv[i]:>8.2f} {bf[i]:>8.2f} "
              f"{bv[i] - bf[i]:>8.2f} {bin_delta_s[i]:>8.2f}")


if __name__ == "__main__":
    main()
