#!/usr/bin/env python3
"""Optuna augmentation search for wl12000 target_hand 8ch (decoupled v2).

Searches 16 physical-quantity dimensions covering 8 augmentation types. Each
sub-parameter of interest is sampled directly in its physical units (e.g.
mask_prob ∈ [0, 0.8], num_masks ∈ [0, 20], snr_db ∈ [20, 55]) rather than via
an abstract per-aug strength. This decouples previously locked sub-parameters
(e.g. gain prob vs gain magnitude, noise prob vs SNR) and avoids the saturation
problems of the strength-based parameterization.

Uses the wl12000 experiment config as the base (8ch target_hand, filtered_paper),
with per-channel + per-hand normalization.

Search dimensions (16 total):
  gain_prob, gain_min, gain_max        — random_gain rate + multiplicative range
  warp_sigma, warp_prob                — mag_warping magnitude + rate
  drift_prob, drift_amp                — baseline_drift rate + max amplitude
  powerline_prob, powerline_harm       — powerline rate + harmonic count
  channel_mask_p                       — channel_mask rate
  time_num_masks, time_mask_size       — time_mask count + width
  freq_num_masks, freq_mask_size       — freq_mask count + width
  noise_prob, noise_snr                — gaussian_noise rate + min SNR (dB)

Usage:
    python scripts/hparam/optuna_aug_search_wl12000.py \
        --gpus 0,1,2,3,4,5 \
        --n-trials 80 \
        --objective-metric val_mae
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
import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT}/assets/optuna_aug_search_wl12000_normfix.db"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "optuna_aug_search_wl12000_normfix"
EXPERIMENT = "emgformer/regression_egoemg_window_ablation_wl12000"
DATA_DIR = "./data/EgoEMG_memmap"
BATCH_AUG_CONFIG = PROJECT_ROOT / "config" / "augmentation" / "batch_aug.yaml"
NORM_STATS_PATH = (
    PROJECT_ROOT / "assets" / "per_dataset_norm_stats_repro_filtered_paper_alias.json"
)
ZERO_DEFAULT_MAX_PROB = 0.6


# ---------------------------------------------------------------------------
# batch_augmentation helpers
# ---------------------------------------------------------------------------


def _load_batch_aug_defaults() -> dict:
    with BATCH_AUG_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["batch_augmentation"]


def build_overrides(
    *,
    trial_number: int,
    gpus: str,
    max_epochs: int | None,
    trial_dir: str,
    params: dict,
    extra_overrides: list[str] | None = None,
    fixed_seed: int | None = None,
) -> list[str]:
    """Build the emg2pose.train command from a dict of physical parameters.

    ``params`` is a flat dict of the 16 search dimensions (physical quantities,
    not abstract strengths) produced by ``objective``. Sub-parameters not in
    the search space keep their ``batch_aug.yaml`` defaults.

    ``fixed_seed`` (if given) overrides the per-trial seed so that every trial
    shares the same weight-init / data-shuffle randomness. This isolates the
    effect of the augmentations from seed-driven variance.
    """
    defaults = _load_batch_aug_defaults()
    seed = fixed_seed if fixed_seed is not None else trial_number
    p = dict(params)  # shallow copy so we can enforce constraints in place

    # Constraint: gain_min must be < gain_max. Swap if violated so the trial
    # stays usable (the sampler may sample min > max on the flat space).
    if p["gain_min"] >= p["gain_max"]:
        p["gain_min"], p["gain_max"] = p["gain_max"], p["gain_min"]
        if p["gain_max"] - p["gain_min"] < 0.01:
            p["gain_max"] = min(1.0, p["gain_min"] + 0.01)

    # Sub-params kept at batch_aug.yaml defaults (not searched). Amplitudes
    # that are not independently searched keep the default max; their min is 0.
    warp_num_knots = int(defaults["mag_warping"]["num_knots"])
    drift_min_freq = float(defaults["baseline_drift"]["min_freq"])
    drift_max_freq = float(defaults["baseline_drift"]["max_freq"])
    drift_min_amp = 0.0
    pl_min_amp = 0.0
    pl_max_amp = float(defaults["powerline_noise"]["max_amp_ratio"])
    gn_max_snr = float(defaults["gaussian_noise"]["max_snr_db"])

    overrides = [
        f"experiment={EXPERIMENT}",
        f"egoemg_memmap_dir={DATA_DIR}",
        f"trainer.devices=[{gpus}]",
        "+trainer.strategy=ddp",
        "trainer.check_val_every_n_epoch=null",
        "+trainer.val_check_interval=100",
        f"seed={seed}",
        f"hydra.run.dir={trial_dir}",
        # WL=12000, 8ch target_hand — same as wl12000.yaml but enforced via CLI
        "datamodule.window_length=12000",
        "datamodule.val_test_window_length=12000",
        "datamodule.stride=1200",
        "datamodule.val_test_stride=12000",
        "egoemg_emg_layout=target_hand",
        "+egoemg_emg2pose_channel_indices=null",
        f"datamodule.per_dataset_norm_stats_path={NORM_STATS_PATH}",
        # ── Augmentation overrides (physical quantities) ──
        f"batch_augmentation.random_gain.min_gain={p['gain_min']:.4f}",
        f"batch_augmentation.random_gain.max_gain={p['gain_max']:.4f}",
        f"batch_augmentation.random_gain.mask_prob={p['gain_prob']:.4f}",
        f"batch_augmentation.mag_warping.sigma={p['warp_sigma']:.4f}",
        f"batch_augmentation.mag_warping.num_knots={warp_num_knots}",
        f"batch_augmentation.mag_warping.mask_prob={p['warp_prob']:.4f}",
        f"batch_augmentation.baseline_drift.mask_prob={p['drift_prob']:.4f}",
        f"batch_augmentation.baseline_drift.min_freq={drift_min_freq:.4f}",
        f"batch_augmentation.baseline_drift.max_freq={drift_max_freq:.4f}",
        f"batch_augmentation.baseline_drift.min_amp_ratio={drift_min_amp:.4f}",
        f"batch_augmentation.baseline_drift.max_amp_ratio={p['drift_amp']:.4f}",
        f"batch_augmentation.powerline_noise.mask_prob={p['powerline_prob']:.4f}",
        f"batch_augmentation.powerline_noise.min_amp_ratio={pl_min_amp:.4f}",
        f"batch_augmentation.powerline_noise.max_amp_ratio={pl_max_amp:.4f}",
        f"batch_augmentation.powerline_noise.max_harmonic={p['powerline_harm']}",
        f"batch_augmentation.channel_mask.mask_prob={p['channel_mask_p']:.4f}",
        f"batch_augmentation.time_mask.num_masks={p['time_num_masks']}",
        f"batch_augmentation.time_mask.max_mask_size={p['time_mask_size']}",
        f"batch_augmentation.freq_mask.num_masks={p['freq_num_masks']}",
        f"batch_augmentation.freq_mask.max_mask_size={p['freq_mask_size']}",
        f"batch_augmentation.gaussian_noise.min_snr_db={p['noise_snr']:.4f}",
        f"batch_augmentation.gaussian_noise.max_snr_db={gn_max_snr:.4f}",
        f"batch_augmentation.gaussian_noise.apply_prob={p['noise_prob']:.4f}",
    ]
    if max_epochs is not None:
        overrides.append(f"trainer.max_epochs={max_epochs}")
    if extra_overrides:
        overrides.extend(extra_overrides)
    return [sys.executable, "-m", "emg2pose.train", *overrides]


# ---------------------------------------------------------------------------
# Parse metric from training stdout
# ---------------------------------------------------------------------------


def parse_metric(stdout: str, metric_name: str) -> float | None:
    match = re.search(rf"'{re.escape(metric_name)}':\s*([\d.eE+-]+)", stdout)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(
    gpus: str,
    max_epochs: int | None,
    objective_metric: str,
    study_name: str,
    log_root: Path,
    extra_overrides: list[str] | None = None,
    fixed_seed: int | None = None,
):
    def objective(trial: optuna.Trial) -> float:
        # Fully decoupled search space (16 physical-quantity dimensions).
        # Each sub-parameter of interest is sampled directly in its physical
        # units, avoiding the abstract-strength mapping and its saturation
        # problems. Probs are bounded well below 1.0 so the sampler never
        # wastes budget in the saturated region.
        p = {
            # random_gain: prob, min, max (decoupled so gain range & rate vary independently)
            "gain_prob":      trial.suggest_float("gain_prob", 0.0, 0.8),
            "gain_min":       trial.suggest_float("gain_min", 0.1, 0.9),
            "gain_max":       trial.suggest_float("gain_max", 0.1, 1.0),
            # mag_warping: sigma + prob (num_knots kept at default)
            "warp_sigma":     trial.suggest_float("warp_sigma", 0.0, 0.8),
            "warp_prob":      trial.suggest_float("warp_prob", 0.0, 0.8),
            # baseline_drift: prob + max amp (freq kept at default)
            "drift_prob":     trial.suggest_float("drift_prob", 0.0, 1.0),
            "drift_amp":      trial.suggest_float("drift_amp", 0.0, 0.25),
            # powerline_noise: prob + harmonic count (amp kept at default)
            "powerline_prob": trial.suggest_float("powerline_prob", 0.0, 0.8),
            "powerline_harm": trial.suggest_int("powerline_harm", 1, 15),
            # channel_mask: prob (default 0; history showed strong discrimination)
            "channel_mask_p": trial.suggest_float("channel_mask_p", 0.0, 0.5),
            # time_mask: count + size (both searched as direct integers)
            "time_num_masks": trial.suggest_int("time_num_masks", 0, 20),
            "time_mask_size": trial.suggest_int("time_mask_size", 0, 1500),
            # freq_mask: count + size
            "freq_num_masks": trial.suggest_int("freq_num_masks", 0, 15),
            "freq_mask_size": trial.suggest_int("freq_mask_size", 0, 400),
            # gaussian_noise: prob + min SNR (decoupled; prob no longer saturates)
            "noise_prob":     trial.suggest_float("noise_prob", 0.0, 1.0),
            "noise_snr":      trial.suggest_float("noise_snr", 20.0, 55.0),
        }
        # Note: the gain_min < gain_max constraint is enforced inside
        # build_overrides so it applies to every entry point.

        # Trial dir
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trial_dir = str(log_root / f"trial_{trial.number:04d}_{ts}")

        cmd = build_overrides(
            trial_number=trial.number, gpus=gpus, max_epochs=max_epochs,
            trial_dir=trial_dir, params=p,
            extra_overrides=extra_overrides,
            fixed_seed=fixed_seed,
        )

        log.info("Trial %d — %s", trial.number, " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                capture_output=True, text=True)

        if result.returncode != 0:
            log.error("Trial %d FAILED (rc=%d). stderr tail:\n%s",
                      trial.number, result.returncode,
                      textwrap.indent("\n".join(result.stderr.strip().splitlines()[-30:]), "    "))
            return float("inf")

        metric_value = parse_metric(result.stdout, objective_metric)
        if metric_value is None:
            log.error("Trial %d — could not parse %s. stdout tail:\n%s",
                      trial.number,
                      objective_metric,
                      textwrap.indent("\n".join(result.stdout.strip().splitlines()[-20:]), "    "))
            return float("inf")

        log.info("Trial %d — %s = %.6f | seed=%s | %s",
                 trial.number, objective_metric, metric_value,
                 fixed_seed if fixed_seed is not None else trial.number,
                 ", ".join(f"{k}={v:.3f}" for k, v in p.items()))
        return metric_value

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna aug search (16 decoupled physical-quantity dims) "
                    "for wl12000 target_hand 8ch"
    )
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument(
        "--n-trials",
        type=int,
        default=80,
        help="Number of trials (default: 80). The 16-D search space benefits "
             "from more trials than the prior 8-D strength search.",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Per-trial epoch budget (default: 100). Pass -1 to use the "
             "experiment config default (150).",
    )
    p.add_argument("--storage", default=DEFAULT_STORAGE)
    p.add_argument("--study-name", default="aug-decoupled-wl12000-perch-v2")
    p.add_argument(
        "--objective-metric",
        choices=["val_user_mae", "val_mae"],
        default="val_user_mae",
        help="Metric parsed from training stdout and minimized by Optuna.",
    )
    p.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help=(
            "Directory for per-trial logs. Defaults to "
            "logs/optuna_aug_search_wl12000_normfix for val_user_mae and "
            "logs/optuna_aug_search_wl12000_normfix_val_mae for val_mae."
        ),
    )
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument(
        "--fixed-seed",
        type=int,
        default=42,
        help=(
            "Use the same seed for every trial so the only varying factor is "
            "the augmentation strength. Pass -1 to restore the legacy "
            "per-trial seed (seed=trial.number). Default: 42."
        ),
    )
    p.add_argument("--extra-override", action="append",
                   dest="extra_overrides", default=[])
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
    log.info("Objective metric: %s", args.objective_metric)
    log_root = args.log_root
    if log_root is None:
        log_root = (
            DEFAULT_LOG_ROOT
            if args.objective_metric == "val_user_mae"
            else PROJECT_ROOT / "logs" / "optuna_aug_search_wl12000_normfix_val_mae"
        )
    if not log_root.is_absolute():
        log_root = PROJECT_ROOT / log_root
    log_root.mkdir(parents=True, exist_ok=True)
    log.info("Trial log root: %s", log_root)

    # Resolve the per-trial seed policy. --fixed-seed -1 restores the legacy
    # behaviour of seed=trial.number (different seed per trial), which mixes
    # seed-driven variance into the augmentation-strength signal.
    fixed_seed = args.fixed_seed if args.fixed_seed >= 0 else None
    if fixed_seed is not None:
        log.info("Per-trial seed: FIXED at %d (all trials share one seed).", fixed_seed)
    else:
        log.info("Per-trial seed: trial.number (legacy mode, --fixed-seed -1).")

    # No warm-start: the legacy search used strength-based parameters under
    # scalar normalization. This search uses decoupled physical quantities
    # under per-channel normalization, so the old "best" point is meaningless.
    # Let TPE explore the new 16-D space from scratch.
    log.info("Warm-start disabled (searching decoupled physical-quantity space).")

    # Map --max-epochs -1 → None so the experiment config default is used.
    max_epochs = args.max_epochs if args.max_epochs and args.max_epochs > 0 else None
    if max_epochs is not None:
        log.info("Per-trial budget: %d epochs.", max_epochs)
    else:
        log.info("Per-trial budget: experiment config default (150 epochs).")

    objective = make_objective(
        gpus=args.gpus, max_epochs=max_epochs,
        objective_metric=args.objective_metric,
        study_name=args.study_name,
        log_root=log_root,
        extra_overrides=args.extra_overrides or None,
        fixed_seed=fixed_seed,
    )

    study.optimize(objective, n_trials=args.n_trials)

    if not any(t.value is not None for t in study.trials):
        log.info("No completed trials yet.")
        return

    best = study.best_trial
    log.info("===== Best trial: %d =====", best.number)
    log.info("  %s = %.6f", args.objective_metric, best.value)
    log.info("  params:")
    for k, v in best.params.items():
        log.info("    %s = %s", k, v)


if __name__ == "__main__":
    main()
