#!/usr/bin/env python3
"""Optuna full augmentation search targeting val_user_mae (held-out user generalization).

Searches all 8 augmentation types + lr + dropout from batch_aug.yaml defaults.
Uses regression_egoemg_clean base config (8ch target_hand, EgoEMG-only).

Usage:
    python scripts/hparam/optuna_user_mae_search.py \
        --gpus 0,1,2,3,4,5 \
        --n-trials 60 \
        --max-epochs 10
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT}/assets/optuna_user_mae.db"
EXPERIMENT = "emgformer/regression_egoemg_clean"
DATA_DIR = "./data/EgoEMG_memmap"

# batch_aug.yaml defaults as warm-start baseline
BATCH_AUG_DEFAULTS = {
    "lr": 0.0001,
    "dropout": 0.15,
    "gain_min_gain": 0.5291,
    "gain_range": 0.1448,
    "gain_mask_prob": 0.1422,
    "warp_sigma": 0.1770,
    "warp_num_knots": 8,
    "warp_mask_prob": 0.0,
    "drift_mask_prob": 0.3544,
    "drift_min_freq": 0.0114,
    "drift_freq_range": 0.4511,
    "drift_min_amp": 0.0154,
    "drift_amp_range": 0.0418,
    "powerline_mask_prob": 0.0800,
    "powerline_min_amp": 0.0141,
    "powerline_amp_range": 0.0487,
    "powerline_max_harmonic": 3,
    "channel_mask_prob": 0.0,
    "time_num_masks": 6,
    "freq_num_masks": 4,
    "noise_min_snr_db": 44.0937,
    "noise_snr_range_db": 5.9063,
    "noise_apply_prob": 0.7761,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_command(
    *,
    trial_number: int,
    gpus: str,
    max_epochs: int,
    trial_dir: str,
    # ── Optimization ──
    lr: float,
    dropout: float,
    # ── RandomGain ──
    gain_min_gain: float,
    gain_max_gain: float,
    gain_mask_prob: float,
    # ── MagWarping ──
    warp_sigma: float,
    warp_num_knots: int,
    warp_mask_prob: float,
    # ── BaselineDrift ──
    drift_mask_prob: float,
    drift_min_freq: float,
    drift_max_freq: float,
    drift_min_amp: float,
    drift_max_amp: float,
    # ── PowerlineNoise ──
    powerline_mask_prob: float,
    powerline_min_amp: float,
    powerline_max_amp: float,
    powerline_max_harmonic: int,
    # ── ChannelMask ──
    channel_mask_prob: float,
    # ── TimeMask ──
    time_num_masks: int,
    # ── FreqMask ──
    freq_num_masks: int,
    # ── GaussianNoise ──
    noise_min_snr_db: float,
    noise_max_snr_db: float,
    noise_apply_prob: float,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    overrides = [
        f"experiment={EXPERIMENT}",
        f"egoemg_memmap_dir={DATA_DIR}",
        f"trainer.devices=[{gpus}]",
        "+trainer.strategy=ddp",
        f"trainer.max_epochs={max_epochs}",
        "trainer.check_val_every_n_epoch=null",
        "+trainer.val_check_interval=10",
        f"seed={trial_number}",
        f"hydra.run.dir={trial_dir}",
        "module.featurizer.conv_blocks.0.in_channels=8",
        f"optimizer.lr={lr:.6f}",
        f"module.decoder.dropout={dropout:.3f}",
        # ── RandomGain ──
        f"batch_augmentation.random_gain.min_gain={gain_min_gain:.4f}",
        f"batch_augmentation.random_gain.max_gain={gain_max_gain:.4f}",
        f"batch_augmentation.random_gain.mask_prob={gain_mask_prob:.4f}",
        # ── MagWarping ──
        f"batch_augmentation.mag_warping.sigma={warp_sigma:.4f}",
        f"batch_augmentation.mag_warping.num_knots={warp_num_knots}",
        f"batch_augmentation.mag_warping.mask_prob={warp_mask_prob:.4f}",
        # ── BaselineDrift ──
        f"batch_augmentation.baseline_drift.mask_prob={drift_mask_prob:.4f}",
        f"batch_augmentation.baseline_drift.min_freq={drift_min_freq:.4f}",
        f"batch_augmentation.baseline_drift.max_freq={drift_max_freq:.4f}",
        f"batch_augmentation.baseline_drift.min_amp_ratio={drift_min_amp:.4f}",
        f"batch_augmentation.baseline_drift.max_amp_ratio={drift_max_amp:.4f}",
        # ── PowerlineNoise ──
        f"batch_augmentation.powerline_noise.mask_prob={powerline_mask_prob:.4f}",
        f"batch_augmentation.powerline_noise.min_amp_ratio={powerline_min_amp:.4f}",
        f"batch_augmentation.powerline_noise.max_amp_ratio={powerline_max_amp:.4f}",
        f"batch_augmentation.powerline_noise.max_harmonic={powerline_max_harmonic}",
        # ── ChannelMask ──
        f"batch_augmentation.channel_mask.mask_prob={channel_mask_prob:.4f}",
        # ── TimeMask ──
        f"batch_augmentation.time_mask.num_masks={time_num_masks}",
        # ── FreqMask ──
        f"batch_augmentation.freq_mask.num_masks={freq_num_masks}",
        # ── GaussianNoise ──
        f"batch_augmentation.gaussian_noise.min_snr_db={noise_min_snr_db:.4f}",
        f"batch_augmentation.gaussian_noise.max_snr_db={noise_max_snr_db:.4f}",
        f"batch_augmentation.gaussian_noise.apply_prob={noise_apply_prob:.4f}",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)
    return [sys.executable, "-m", "emg2pose.train", *overrides]


def parse_user_mae(stdout: str) -> float | None:
    match = re.search(r"'val_user_mae':\s*([\d.eE+-]+)", stdout)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def make_objective(
    gpus: str,
    max_epochs: int,
    study_name: str,
    extra_overrides: list[str] | None = None,
):
    def objective(trial: optuna.Trial) -> float:
        # ── Optimization ────────────────────────────────────────────────
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.45)

        # ── RandomGain ──────────────────────────────────────────────────
        gain_min_gain = trial.suggest_float("gain_min_gain", 0.5, 0.95)
        gain_range = trial.suggest_float("gain_range", 0.02, 0.5)
        gain_max_gain = gain_min_gain + gain_range
        gain_mask_prob = trial.suggest_float("gain_mask_prob", 0.0, 1.0)
        trial.set_user_attr("gain_max_gain", gain_max_gain)

        # ── MagWarping ──────────────────────────────────────────────────
        warp_sigma = trial.suggest_float("warp_sigma", 0.02, 0.3)
        warp_num_knots = trial.suggest_int("warp_num_knots", 4, 16)
        warp_mask_prob = trial.suggest_float("warp_mask_prob", 0.0, 1.0)

        # ── BaselineDrift ───────────────────────────────────────────────
        drift_mask_prob = trial.suggest_float("drift_mask_prob", 0.0, 0.9)
        drift_min_freq = trial.suggest_float("drift_min_freq", 0.01, 0.15)
        drift_freq_range = trial.suggest_float("drift_freq_range", 0.05, 0.8)
        drift_max_freq = drift_min_freq + drift_freq_range
        drift_min_amp = trial.suggest_float("drift_min_amp", 0.002, 0.05)
        drift_amp_range = trial.suggest_float("drift_amp_range", 0.01, 0.12)
        drift_max_amp = drift_min_amp + drift_amp_range
        trial.set_user_attr("drift_max_freq", drift_max_freq)
        trial.set_user_attr("drift_max_amp", drift_max_amp)

        # ── PowerlineNoise ──────────────────────────────────────────────
        powerline_mask_prob = trial.suggest_float("powerline_mask_prob", 0.0, 0.5)
        powerline_min_amp = trial.suggest_float("powerline_min_amp", 0.001, 0.02)
        powerline_amp_range = trial.suggest_float("powerline_amp_range", 0.005, 0.05)
        powerline_max_amp = powerline_min_amp + powerline_amp_range
        powerline_max_harmonic = trial.suggest_int("powerline_max_harmonic", 1, 5)
        trial.set_user_attr("powerline_max_amp", powerline_max_amp)

        # ── ChannelMask ─────────────────────────────────────────────────
        channel_mask_prob = trial.suggest_float("channel_mask_prob", 0.0, 0.5)

        # ── TimeMask ────────────────────────────────────────────────────
        time_num_masks = trial.suggest_int("time_num_masks", 0, 20)

        # ── FreqMask ────────────────────────────────────────────────────
        freq_num_masks = trial.suggest_int("freq_num_masks", 0, 20)

        # ── GaussianNoise ───────────────────────────────────────────────
        noise_min_snr_db = trial.suggest_float("noise_min_snr_db", 10.0, 45.0)
        noise_snr_range_db = trial.suggest_float("noise_snr_range_db", 2.0, 25.0)
        noise_max_snr_db = min(noise_min_snr_db + noise_snr_range_db, 50.0)
        noise_apply_prob = trial.suggest_float("noise_apply_prob", 0.1, 1.0)
        trial.set_user_attr("noise_max_snr_db", noise_max_snr_db)

        # ── Trial directory ─────────────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trial_dir = os.path.join(
            PROJECT_ROOT, "logs", "optuna_user_mae",
            f"trial_{trial.number:04d}_{ts}",
        )

        cmd = build_command(
            trial_number=trial.number, gpus=gpus, max_epochs=max_epochs,
            trial_dir=trial_dir, lr=lr, dropout=dropout,
            gain_min_gain=gain_min_gain, gain_max_gain=gain_max_gain,
            gain_mask_prob=gain_mask_prob,
            warp_sigma=warp_sigma, warp_num_knots=warp_num_knots,
            warp_mask_prob=warp_mask_prob,
            drift_mask_prob=drift_mask_prob, drift_min_freq=drift_min_freq,
            drift_max_freq=drift_max_freq, drift_min_amp=drift_min_amp,
            drift_max_amp=drift_max_amp,
            powerline_mask_prob=powerline_mask_prob,
            powerline_min_amp=powerline_min_amp,
            powerline_max_amp=powerline_max_amp,
            powerline_max_harmonic=powerline_max_harmonic,
            channel_mask_prob=channel_mask_prob,
            time_num_masks=time_num_masks, freq_num_masks=freq_num_masks,
            noise_min_snr_db=noise_min_snr_db, noise_max_snr_db=noise_max_snr_db,
            noise_apply_prob=noise_apply_prob,
            extra_overrides=extra_overrides,
        )

        log.info("Trial %d — cmd: %s", trial.number, " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                capture_output=True, text=True)

        if result.returncode != 0:
            log.error("Trial %d FAILED (rc=%d). stderr:\n%s",
                      trial.number, result.returncode,
                      textwrap.indent("\n".join(result.stderr.strip().splitlines()[-30:]), "    "))
            return float("inf")

        user_mae = parse_user_mae(result.stdout)
        if user_mae is None:
            log.error("Trial %d — could not parse val_user_mae. stdout tail:\n%s",
                      trial.number,
                      textwrap.indent("\n".join(result.stdout.strip().splitlines()[-20:]), "    "))
            return float("inf")

        log.info("Trial %d — val_user_mae = %.6f", trial.number, user_mae)
        return user_mae

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna full augmentation search targeting val_user_mae"
    )
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--storage", default=DEFAULT_STORAGE)
    p.add_argument("--study-name", default="egoemg-user-mae-v2")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--extra-override", action="append",
                   dest="extra_overrides", default=[])
    p.add_argument("--no-enqueue", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    sampler = (
        optuna.samplers.TPESampler(seed=args.sampler_seed)
        if args.sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=args.sampler_seed)
    )

    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        direction="minimize", sampler=sampler, load_if_exists=True,
    )

    log.info("Study %r — %d existing trials.", args.study_name, len(study.trials))

    # Enqueue batch_aug defaults as warm start
    if len(study.trials) == 0:
        if not args.no_enqueue:
            study.enqueue_trial(BATCH_AUG_DEFAULTS)
            log.info("Enqueued batch_aug.yaml defaults as warm start.")

    objective = make_objective(
        gpus=args.gpus, max_epochs=args.max_epochs,
        study_name=args.study_name,
        extra_overrides=args.extra_overrides or None,
    )

    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    log.info("===== Best trial: %d =====", best.number)
    log.info("  val_user_mae = %.6f", best.value)
    log.info("  params:")
    for k, v in best.params.items():
        log.info("    %s = %s", k, v)


if __name__ == "__main__":
    main()
