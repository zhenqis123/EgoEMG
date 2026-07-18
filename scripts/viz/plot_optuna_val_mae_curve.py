#!/usr/bin/env python3
"""Plot val_mae vs. trial number for the optuna augmentation-strength sweep.

Reads the optuna sqlite DB and renders:
  1. per-trial val_mae (scatter + line)
  2. cumulative best-so-far val_mae (highlighting convergence)
  3. annotations for the overall best trial

Usage:
  python scripts/viz/plot_optuna_val_mae_curve.py
  python scripts/viz/plot_optuna_val_mae_curve.py --db PATH --out PATH
"""
from __future__ import annotations

import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DB = "assets/optuna_aug_search_wl12000_normfix_val_mae.db"
DEFAULT_OUT = "figures/optuna_val_mae_curve.png"


def load_trials(db_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (trial_numbers, val_mae) ordered by trial number."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT t.number, tv.value
        FROM trials t
        JOIN trial_values tv ON tv.trial_id = t.rowid
        WHERE t.state = 'COMPLETE' AND tv.value IS NOT NULL
        ORDER BY t.number
        """
    ).fetchall()
    con.close()
    numbers = np.array([r[0] for r in rows], dtype=float)
    values = np.array([r[1] for r in rows], dtype=float)
    return numbers, values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB, help="Path to optuna sqlite DB")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output image path")
    args = ap.parse_args()

    numbers, values = load_trials(args.db)
    best_idx = int(np.argmin(values))
    best_num = int(numbers[best_idx])
    best_val = float(values[best_idx])
    running_min = np.minimum.accumulate(values)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=130)

    # Per-trial val_mae
    ax.plot(
        numbers, values, color="#9aa6b2", linewidth=1.3,
        marker="o", markersize=5, markerfacecolor="#5b6b7f",
        markeredgecolor="white", markeredgewidth=0.5,
        label="val_mae (per trial)", zorder=2,
    )

    # Cumulative best-so-far
    ax.plot(
        numbers, running_min, color="#d64545", linewidth=2.4,
        label="Best-so-far val_mae", zorder=4,
    )

    # Annotate global best
    ax.scatter(
        [best_num], [best_val], s=140, facecolor="none",
        edgecolor="#d64545", linewidth=2.2, zorder=5,
    )
    ax.annotate(
        f"Best: trial {best_num}\nval_mae = {best_val:.4f}",
        xy=(best_num, best_val),
        xytext=(best_num + 1.5, best_val - 0.0025),
        fontsize=10, color="#d64545", weight="bold",
        arrowprops=dict(arrowstyle="->", color="#d64545", lw=1.2),
        zorder=6,
    )

    ax.set_xlabel("Trial number", fontsize=12)
    ax.set_ylabel("val_mae", fontsize=12)
    ax.set_title(
        "Optuna Augmentation-Strength Sweep — val_mae over trials\n"
        f"(WL=12000, {len(numbers)} trials, study: aug-strength-wl12000-normfix-val-mae-v1)",
        fontsize=13,
    )
    ax.set_xticks(np.arange(0, int(numbers.max()) + 1, 5))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)

    # Tighten y range around the data for readability
    ymin, ymax = float(values.min()), float(values.max())
    pad = (ymax - ymin) * 0.12
    ax.set_ylim(ymin - pad * 1.5, ymax + pad)
    ax.set_xlim(-1, int(numbers.max()) + 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}  ({len(numbers)} trials, best=val_mae {best_val:.6f} @ trial {best_num})")


if __name__ == "__main__":
    main()
