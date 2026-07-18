#!/usr/bin/env python3
"""V3 augmentation search: focused on freq_mask + drift + mixup + noise.

Drops confirmed-harmful params (rotation, time_mask) and extends the
freq_num_masks range beyond V2's ceiling. Adds freq_mask_size and noise_snr
as new search dimensions. Base config locked to V2 best trial #12
(val_mae=0.2446).

Search dimensions (7):
  freq_num_masks   [14, 30]   — extended from V2's [8,18] (best was at ceiling)
  freq_mask_size   [20, 80]   — NEW: mask width interaction with count
  drift_prob       [0.4, 0.9]
  drift_amp        [0.08, 0.25] — nudged upper bound from 0.20
  mixup_prob       [0.0, 0.5]
  mixup_alpha      [0.15, 0.35] — narrowed from [0.1,0.4]
  noise_snr        [40, 50]   — NEW: re-explore noise floor

Usage:
    python scripts/hparam/aug_v3_search.py --gpus 2,3,4,5 --n-trials 19
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import optuna

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT}/assets/aug_v3_search.db"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "aug_v3_search"
EXPERIMENT = "emgformer/regression_egoemg_window_ablation_wl12000"
DATA_DIR = "/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap"
NORM_STATS_PATH = (
    PROJECT_ROOT / "assets" / "per_dataset_norm_stats_repro_filtered_paper_alias.json"
)

# Base config locked to V2 best trial #12 (val_mae=0.2446).
# Only the 7 searched params above are varied; everything else is fixed here.
BASE_CONFIG_OVERRIDES = [
    # random_gain (trial 54/12 — effectively off)
    "batch_augmentation.random_gain.min_gain=0.1996",
    "batch_augmentation.random_gain.max_gain=0.9486",
    "batch_augmentation.random_gain.mask_prob=0.0295",
    # mag_warping (off)
    "batch_augmentation.mag_warping.sigma=0.7302",
    "batch_augmentation.mag_warping.num_knots=15",
    "batch_augmentation.mag_warping.mask_prob=0.0049",
    # baseline_drift fixed sub-params (prob & amp are searched)
    "batch_augmentation.baseline_drift.min_freq=0.0114",
    "batch_augmentation.baseline_drift.max_freq=0.4625",
    "batch_augmentation.baseline_drift.min_amp_ratio=0.0",
    # powerline_noise (fixed from trial 54)
    "batch_augmentation.powerline_noise.mask_prob=0.2413",
    "batch_augmentation.powerline_noise.min_amp_ratio=0.0",
    "batch_augmentation.powerline_noise.max_amp_ratio=0.0628",
    "batch_augmentation.powerline_noise.max_harmonic=7",
    # channel_mask (off)
    "batch_augmentation.channel_mask.mask_prob=0.0016",
    # time_mask (off — confirmed harmful in V2)
    "batch_augmentation.time_mask.num_masks=0",
    "batch_augmentation.time_mask.max_mask_size=243",
    "+batch_augmentation.time_mask.per_channel=true",
    # channel_rotation (off — confirmed harmful for 8ch)
    "+batch_augmentation.channel_rotation.mask_prob=0.0",
    "+batch_augmentation.channel_rotation.max_shift=2",
    # gaussian_noise fixed sub-params (snr is searched, prob fixed)
    "batch_augmentation.gaussian_noise.apply_prob=0.1126",
    "batch_augmentation.gaussian_noise.max_snr_db=50.0",
    # mixup base (prob & alpha searched via separate keys below)
]


def parse_metric(stdout: str, metric_name: str) -> float | None:
    match = re.search(rf"'{re.escape(metric_name)}':\s*([\d.eE+-]+)", stdout)
    return float(match.group(1)) if match else None


def make_objective(gpus, max_epochs, log_root, fixed_seed):
    def objective(trial: optuna.Trial) -> float:
        # ── 7 searched params ────────────────────────────────────────
        freq_num_masks = trial.suggest_int("freq_num_masks", 14, 30)
        freq_mask_size = trial.suggest_int("freq_mask_size", 20, 80)
        drift_prob = trial.suggest_float("drift_prob", 0.4, 0.9)
        drift_amp = trial.suggest_float("drift_amp", 0.08, 0.25)
        mixup_prob = trial.suggest_float("mixup_prob", 0.0, 0.5)
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.15, 0.35)
        noise_snr = trial.suggest_float("noise_snr", 40.0, 50.0)

        p = {
            "freq_num_masks": freq_num_masks,
            "freq_mask_size": freq_mask_size,
            "drift_prob": drift_prob,
            "drift_amp": drift_amp,
            "mixup_prob": mixup_prob,
            "mixup_alpha": mixup_alpha,
            "noise_snr": noise_snr,
        }

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trial_dir = str(log_root / f"trial_{trial.number:04d}_{ts}")
        seed = fixed_seed if fixed_seed is not None else trial.number

        cmd = [
            sys.executable, "-m", "emg2pose.train",
            f"experiment={EXPERIMENT}",
            f"egoemg_memmap_dir={DATA_DIR}",
            f"trainer.devices=[{gpus}]",
            "+trainer.strategy=ddp",
            "trainer.check_val_every_n_epoch=null",
            "+trainer.val_check_interval=100",
            f"seed={seed}",
            f"hydra.run.dir={trial_dir}",
            "datamodule.window_length=12000",
            "datamodule.val_test_window_length=12000",
            "datamodule.stride=1200",
            "datamodule.val_test_stride=12000",
            "egoemg_emg_layout=target_hand",
            "+egoemg_emg2pose_channel_indices=null",
            f"datamodule.per_dataset_norm_stats_path={NORM_STATS_PATH}",
            f"trainer.max_epochs={max_epochs}",
            # Base config (trial 12 fixed)
            *BASE_CONFIG_OVERRIDES,
            # Searched params
            f"batch_augmentation.freq_mask.num_masks={freq_num_masks}",
            f"batch_augmentation.freq_mask.max_mask_size={freq_mask_size}",
            f"batch_augmentation.baseline_drift.mask_prob={drift_prob:.4f}",
            f"batch_augmentation.baseline_drift.max_amp_ratio={drift_amp:.4f}",
            f"+batch_augmentation.mixup.mask_prob={mixup_prob:.4f}",
            f"+batch_augmentation.mixup.alpha={mixup_alpha:.4f}",
            f"batch_augmentation.gaussian_noise.min_snr_db={noise_snr:.4f}",
        ]

        log.info("Trial %d — %s", trial.number, " ".join(cmd))
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True
        )
        if result.returncode != 0:
            log.error("Trial %d FAILED (rc=%d). stderr tail:\n%s",
                      trial.number, result.returncode,
                      textwrap.indent("\n".join(result.stderr.strip().splitlines()[-30:]), "    "))
            return float("inf")

        metric_value = parse_metric(result.stdout, "val_mae")
        if metric_value is None:
            log.error("Trial %d — could not parse val_mae", trial.number)
            return float("inf")

        log.info("Trial %d — val_mae = %.6f | seed=%d | %s",
                 trial.number, metric_value, seed,
                 ", ".join(f"{k}={v:.3f}" for k, v in p.items()))
        return metric_value

    return objective


def main():
    p = argparse.ArgumentParser(description="V3 augmentation search (freq+drift+mixup+noise)")
    p.add_argument("--gpus", default="2,3,4,5")
    p.add_argument("--n-trials", type=int, default=19)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--storage", default=DEFAULT_STORAGE)
    p.add_argument("--study-name", default="aug-v3-freq-drift-mixup")
    p.add_argument("--fixed-seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        direction="minimize", sampler=sampler, load_if_exists=True,
    )
    log.info("Study %r — %d existing trials.", args.study_name, len(study.trials))

    log_root = DEFAULT_LOG_ROOT
    log_root.mkdir(parents=True, exist_ok=True)

    fixed_seed = args.fixed_seed if args.fixed_seed >= 0 else None
    log.info("Per-trial seed: %s", fixed_seed if fixed_seed is not None else "trial.number")
    log.info("Base config: V2 best trial #12 (val_mae=0.2446)")
    log.info("Search dims: 7 (freq_num_masks, freq_mask_size, drift_prob, drift_amp, mixup_prob, mixup_alpha, noise_snr)")

    objective = make_objective(args.gpus, args.max_epochs, log_root, fixed_seed)
    study.optimize(objective, n_trials=args.n_trials)

    if study.best_trial:
        log.info("===== Best trial: %d =====", study.best_trial.number)
        log.info("  val_mae = %.6f", study.best_trial.value)
        for k, v in study.best_trial.params.items():
            log.info("    %s = %s", k, v)


if __name__ == "__main__":
    main()
