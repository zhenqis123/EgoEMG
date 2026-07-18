#!/usr/bin/env python3
"""Plot MANO beta parameter distributions to show hand shape diversity.

EgoEmg has per-episode MANO beta (10-dim shape parameters) for each subject.
EMG2Pose uses a single generic_hand_model.json — zero beta diversity.

This script plots EgoEmg's beta distributions and saves the figure.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BETA_DIM_NAMES = [
    "β₁ (Scale)",
    "β₂ (Scale)",
    "β₃ (Finger thickness)",
    "β₄ (Finger thickness)",
    "β₅ (Palm thickness)",
    "β₆ (Palm breadth)",
    "β₇ (Palm breadth)",
    "β₈ (Finger length)",
    "β₉ (Finger length)",
    "β₁₀ (Finger length)",
]


def load_egoemg_betas(memmap_dir):
    mm_dir = Path(memmap_dir)
    manifest = json.loads((mm_dir / "manifest.json").read_text())
    metadata = np.load(mm_dir / "metadata.npz", allow_pickle=True)

    left_info = manifest["episode_fields"]["generated_mano_left_beta"]
    right_info = manifest["episode_fields"]["generated_mano_right_beta"]
    left_beta = np.memmap(
        mm_dir / left_info["filename"],
        dtype=np.dtype(left_info["dtype"]),
        mode="r",
        shape=tuple(left_info["shape"]),
    )
    right_beta = np.memmap(
        mm_dir / right_info["filename"],
        dtype=np.dtype(right_info["dtype"]),
        mode="r",
        shape=tuple(right_info["shape"]),
    )
    episode_ids = metadata["episode_id"]
    return (
        np.array(left_beta),
        np.array(right_beta),
        [e.decode() if isinstance(e, bytes) else str(e) for e in episode_ids],
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--egoemg-dir", type=str, default="data/EgoEMG_memmap")
    parser.add_argument("--output", type=str, default="paper/figures/fig_beta_diversity.pdf")
    args = parser.parse_args()

    print("Loading EgoEmg MANO betas...", file=sys.stderr)
    left_beta, right_beta, episode_ids = load_egoemg_betas(args.egoemg_dir)
    n_episodes, n_dims = left_beta.shape
    print(f"  Loaded {n_episodes} episodes × {n_dims} beta dims", file=sys.stderr)

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, n_episodes))

    for dim_idx, ax in enumerate(axes.flat):
        # Plot left and right betas per episode
        x = np.arange(n_episodes)
        ax.bar(x - 0.15, left_beta[:, dim_idx], 0.3, color="steelblue", alpha=0.8, label="Left")
        ax.bar(x + 0.15, right_beta[:, dim_idx], 0.3, color="darkorange", alpha=0.8, label="Right")

        # Stats
        all_vals = np.concatenate([left_beta[:, dim_idx], right_beta[:, dim_idx]])
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(BETA_DIM_NAMES[dim_idx], fontsize=9)
        ax.set_xticks([])
        ax.tick_params(labelsize=7)

        if dim_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"MANO Shape Parameter Diversity — EgoEmg ({n_episodes} participants, 41 left + 41 right hands)\n"
        f"EMG2Pose: single generic hand model → all 10 β dims ≡ 0 for all subjects",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.output}", file=sys.stderr)

    # Print summary stats
    print("\nPer-dimension statistics (left + right):")
    all_beta = np.concatenate([left_beta, right_beta], axis=0)
    print(f"{'Dimension':<25s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Range':>8s}")
    print("-" * 70)
    for dim_idx, name in enumerate(BETA_DIM_NAMES):
        vals = all_beta[:, dim_idx]
        print(f"{name:<25s} {vals.mean():8.4f} {vals.std():8.4f} {vals.min():8.4f} {vals.max():8.4f} {vals.ptp():8.4f}")


if __name__ == "__main__":
    main()
