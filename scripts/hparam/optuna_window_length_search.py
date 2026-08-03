#!/usr/bin/env python3
"""Optuna window-length search for EgoEMG (middle model) with fixed v5 augmentation.

Fixed batch augmentation from ``config/augmentation/batch_aug.yaml``.
Searches over integer ``window_length`` around 7790.
Stride = window_length // 10, val_test_stride = window_length.
Batch size scales inversely with window_length to keep memory constant
(baseline: bs=500 @ wl=7790).

Usage::

    python scripts/hparam/optuna_window_length_search.py \\
        --gpus 0,1,2,3,4,5 \\
        --n-trials 10 \\
        --max-epochs 150
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import optuna

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT}/assets/optuna_window_length.db"

EXPERIMENT = "emgformer/regression_emgformer_middle_aug_search_egoemg"

# Baseline: bs=500 at wl=7790 fits safely in 24GB.
# Scale inversely so bs * wl stays roughly constant.
_BASELINE_WL = 7790
_BASELINE_BS = 500
_MEM_BUDGET = _BASELINE_WL * _BASELINE_BS  # 3,895,000


def compute_batch_size(window_length: int) -> int:
    """Batch size scaled to keep memory within budget.

    - Long windows (wl > baseline): use bs ∝ (1/wl)^1.5 to account for
      transformer O(T²) attention overhead.
    - Short windows (wl <= baseline): use linear scaling, since attention
      shrink helps proportionally.
    """
    if window_length <= _BASELINE_WL:
        bs = _MEM_BUDGET // window_length
    else:
        ratio = _BASELINE_WL / window_length
        bs = int(_BASELINE_BS * ratio ** 1.5)
    bs = max(50, min(750, bs))
    return bs


def build_command(
    *,
    trial_number: int,
    gpus: str,
    max_epochs: int,
    trial_dir: str,
    egoemg_memmap_dir: str,
    window_length: int,
    batch_size: int,
) -> list[str]:
    stride = max(1, window_length // 10)
    overrides = [
        f"experiment={EXPERIMENT}",
        f"egoemg_memmap_dir={egoemg_memmap_dir}",
        f"trainer.devices=[{gpus}]",
        "+trainer.strategy=ddp",
        f"trainer.max_epochs={max_epochs}",
        f"seed={trial_number}",
        f"hydra.run.dir={trial_dir}",
        f"datamodule.window_length={window_length}",
        f"datamodule.val_test_window_length={window_length}",
        f"datamodule.stride={stride}",
        f"datamodule.val_test_stride={window_length}",
        f"batch_size={batch_size}",
        # fixed v5 augmentation
        "batch_augmentation.random_gain.min_gain=0.5291",
        "batch_augmentation.random_gain.max_gain=0.6739",
        "batch_augmentation.random_gain.mask_prob=0.1422",
        "batch_augmentation.mag_warping.sigma=0.1770",
        "batch_augmentation.mag_warping.num_knots=15",
        "batch_augmentation.mag_warping.mask_prob=0.0253",
        "batch_augmentation.baseline_drift.mask_prob=0.3544",
        "batch_augmentation.baseline_drift.min_freq=0.0114",
        "batch_augmentation.baseline_drift.max_freq=0.4625",
        "batch_augmentation.baseline_drift.min_amp_ratio=0.0154",
        "batch_augmentation.baseline_drift.max_amp_ratio=0.0572",
        "batch_augmentation.powerline_noise.mask_prob=0.0800",
        "batch_augmentation.powerline_noise.min_amp_ratio=0.0141",
        "batch_augmentation.powerline_noise.max_amp_ratio=0.0628",
        "batch_augmentation.powerline_noise.max_harmonic=5",
        "batch_augmentation.channel_mask.mask_prob=0.0112",
        "batch_augmentation.time_mask.num_masks=6",
        "batch_augmentation.time_mask.max_mask_size=500",
        "batch_augmentation.freq_mask.num_masks=4",
        "batch_augmentation.freq_mask.max_mask_size=128",
        "batch_augmentation.gaussian_noise.min_snr_db=44.0937",
        "batch_augmentation.gaussian_noise.max_snr_db=50.0000",
        "batch_augmentation.gaussian_noise.apply_prob=0.7761",
    ]
    return [sys.executable, "-m", "egoemg.train", *overrides]


def parse_val_mae(stdout: str) -> float | None:
    match = re.search(
        r"'val_metrics':\s*\[.*?'val_mae':\s*([\d.eE+-]+)", stdout, re.DOTALL
    )
    if match:
        return float(match.group(1))
    return None


def make_objective(
    gpus: str,
    max_epochs: int,
    egoemg_memmap_dir: str,
    study_name: str,
    wl_min: int = 3000,
    wl_max: int = 15000,
):
    def objective(trial: optuna.Trial) -> float:
        window_length = trial.suggest_int("window_length", wl_min, wl_max)
        batch_size = compute_batch_size(window_length)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trial_dir = os.path.join(
            PROJECT_ROOT,
            "logs",
            "optuna_window",
            study_name,
            f"trial_{trial.number:04d}_{ts}",
        )

        cmd = build_command(
            trial_number=trial.number,
            gpus=gpus,
            max_epochs=max_epochs,
            trial_dir=trial_dir,
            egoemg_memmap_dir=egoemg_memmap_dir,
            window_length=window_length,
            batch_size=batch_size,
        )

        log.info("Trial %d (wl=%d, bs=%d) — launching", trial.number, window_length, batch_size)
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            log.error(
                "Trial %d FAILED (rc=%d). stderr tail:\n%s",
                trial.number,
                result.returncode,
                textwrap.indent(
                    "\n".join(result.stderr.strip().splitlines()[-30:]),
                    "    ",
                ),
            )
            return float("inf")

        val_mae = parse_val_mae(result.stdout)
        if val_mae is None:
            log.error(
                "Trial %d — could not parse val_mae. stdout tail:\n%s",
                trial.number,
                textwrap.indent(
                    "\n".join(result.stdout.strip().splitlines()[-20:]),
                    "    ",
                ),
            )
            return float("inf")

        log.info("Trial %d (wl=%d, bs=%d) — val_mae = %.6f", trial.number, window_length, batch_size, val_mae)
        return val_mae

    return objective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna window-length search for EgoEMG middle model (v5 aug, DDP)"
    )
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--wl-min", type=int, default=3000)
    p.add_argument("--wl-max", type=int, default=15000)
    p.add_argument("--storage", default=DEFAULT_STORAGE)
    p.add_argument("--study-name", default="egoemg-window-v1")
    p.add_argument("--egoemg-memmap-dir", default="./data/EgoEMG_memmap")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--sampler-seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.sampler_seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    log.info("Study %r — %d completed trials already in storage.", args.study_name, len(study.trials))

    objective = make_objective(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        egoemg_memmap_dir=args.egoemg_memmap_dir,
        study_name=args.study_name,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
    )

    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    log.info("===== Best trial: %d =====", best.number)
    log.info("  val_mae = %.6f", best.value)
    log.info("  window_length = %d", best.params["window_length"])


if __name__ == "__main__":
    main()
