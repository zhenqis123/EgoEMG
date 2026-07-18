#!/usr/bin/env python3
"""Extract best val_mae from augmentation sweep experiments."""
from __future__ import annotations

import glob
import os
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_BASE = "logs/ablation/aug_sweep"


def extract_results() -> list[dict]:
    results = []
    for exp_dir in sorted(glob.glob(f"{LOG_BASE}/exp_*")):
        exp_name = os.path.basename(exp_dir)
        # Find version dir
        version_dirs = glob.glob(f"{exp_dir}/regression_egoemg/version_*")
        if not version_dirs:
            print(f"SKIP {exp_name}: no version dir", file=sys.stderr)
            continue
        events = glob.glob(f"{version_dirs[0]}/events.out.*")
        if not events:
            print(f"SKIP {exp_name}: no events file", file=sys.stderr)
            continue

        ea = EventAccumulator(version_dirs[0])
        ea.Reload()

        if "val_mae" not in ea.Tags().get("scalars", []):
            print(f"SKIP {exp_name}: no val_mae", file=sys.stderr)
            continue

        val_mae = [(e.step, e.value) for e in ea.Scalars("val_mae")]
        best_step, best_val = min(val_mae, key=lambda x: x[1])
        last_step, last_val = val_mae[-1]

        # Also get train_loss at the same step
        train_loss = None
        if "train_loss" in ea.Tags().get("scalars", []):
            tl = ea.Scalars("train_loss")
            # Find closest train_loss to best step
            closest = min(tl, key=lambda e: abs(e.step - best_step))
            train_loss = closest.value

        results.append({
            "exp": exp_name,
            "best_step": best_step,
            "best_val_mae": best_val,
            "last_val_mae": last_val,
            "train_loss": train_loss,
        })

    return results


def main():
    results = extract_results()
    if not results:
        print("No results found.")
        return

    # Sort by best val_mae
    results.sort(key=lambda r: r["best_val_mae"])

    print(f"{'Rank':<5} {'Experiment':<45} {'Best Step':>10} {'Best val_mae':>12} {'Last val_mae':>12} {'Train Loss':>10}")
    print("-" * 100)
    for i, r in enumerate(results):
        tl_str = f"{r['train_loss']:.4f}" if r["train_loss"] else "N/A"
        print(f"{i+1:<5} {r['exp']:<45} {r['best_step']:>10} {r['best_val_mae']:>12.6f} {r['last_val_mae']:>12.6f} {tl_str:>10}")


if __name__ == "__main__":
    main()
