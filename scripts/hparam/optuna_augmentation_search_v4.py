#!/usr/bin/env python3
"""Optuna full augmentation hyperparameter search for EgoEMG (middle model) — GPU batch v5.

Uses batched GPU augmentation (BatchAugmentation) with vectorized time/freq masks
and sub-batch extraction for low-prob transforms.  21 params, same ranges as v3.
Top-10 v3 trials enqueued as warm start.

Usage::

    python scripts/hparam/optuna_augmentation_search_v4.py \\
        --gpus 1,2,3,4,5 \\
        --n-trials 60 \\
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

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT}/assets/optuna_augmentation.db"

EXPERIMENT = "emgformer/regression_emgformer_middle_aug_search_egoemg"

# Top-10 v3 trial params enqueued into v5 for warm start (best first)
V3_TOP_TRIALS = [
    {   # Trial #43: val_mae=0.251899
        "gain_min_gain": 0.7147581002171965,
        "gain_range": 0.23359175806972782,
        "gain_mask_prob": 0.01293917193747296,
        "warp_sigma": 0.06549047142430757,
        "warp_num_knots": 4,
        "warp_mask_prob": 0.2258317486523452,
        "drift_mask_prob": 0.19900500136795388,
        "drift_min_freq": 0.05500606880376399,
        "drift_freq_range": 0.07879627351735267,
        "drift_min_amp": 0.009754716613268518,
        "drift_amp_range": 0.05504620493682285,
        "powerline_mask_prob": 0.06926244919503823,
        "powerline_min_amp": 0.01422432314553331,
        "powerline_amp_range": 0.04750947316300547,
        "powerline_max_harmonic": 4,
        "channel_mask_prob": 0.025382831645194087,
        "time_num_masks": 2,
        "freq_num_masks": 7,
        "noise_min_snr_db": 37.21471890643086,
        "noise_snr_range_db": 14.806794692461922,
        "noise_apply_prob": 0.5520679383395867,
    },
    {   # Trial #21: val_mae=0.253472
        "gain_min_gain": 0.7993277957738145,
        "gain_range": 0.18948795943384302,
        "gain_mask_prob": 0.1556566967655723,
        "warp_sigma": 0.09961987164318582,
        "warp_num_knots": 5,
        "warp_mask_prob": 0.4601023819939396,
        "drift_mask_prob": 0.19699354437618516,
        "drift_min_freq": 0.06261264349102079,
        "drift_freq_range": 0.08099184556033662,
        "drift_min_amp": 0.01129487635443934,
        "drift_amp_range": 0.04596226071492245,
        "powerline_mask_prob": 0.10735994057210698,
        "powerline_min_amp": 0.005069163587256587,
        "powerline_amp_range": 0.006129125170722616,
        "powerline_max_harmonic": 2,
        "channel_mask_prob": 0.11988634581035151,
        "time_num_masks": 7,
        "freq_num_masks": 2,
        "noise_min_snr_db": 29.45344779909688,
        "noise_snr_range_db": 20.069500996445794,
        "noise_apply_prob": 0.4982701797366599,
    },
    {   # Trial #44: val_mae=0.254063
        "gain_min_gain": 0.6916086360468785,
        "gain_range": 0.0577952754772306,
        "gain_mask_prob": 0.3383329984697162,
        "warp_sigma": 0.05869869517521643,
        "warp_num_knots": 7,
        "warp_mask_prob": 0.48813822027134556,
        "drift_mask_prob": 0.2668149776538382,
        "drift_min_freq": 0.010125895780058842,
        "drift_freq_range": 0.18345484391773612,
        "drift_min_amp": 0.014614458187487403,
        "drift_amp_range": 0.09669805012442792,
        "powerline_mask_prob": 0.16500953144941484,
        "powerline_min_amp": 0.0016832292786994046,
        "powerline_amp_range": 0.0316449081542192,
        "powerline_max_harmonic": 2,
        "channel_mask_prob": 0.06640438534174358,
        "time_num_masks": 8,
        "freq_num_masks": 4,
        "noise_min_snr_db": 33.179446348403635,
        "noise_snr_range_db": 7.438958270665879,
        "noise_apply_prob": 0.5966457000036058,
    },
    {   # Trial #26: val_mae=0.254079
        "gain_min_gain": 0.7660596057527023,
        "gain_range": 0.22199274012721772,
        "gain_mask_prob": 0.31780255003566855,
        "warp_sigma": 0.06791032182381485,
        "warp_num_knots": 7,
        "warp_mask_prob": 0.3995622275687424,
        "drift_mask_prob": 0.1423409852269057,
        "drift_min_freq": 0.05253945157145088,
        "drift_freq_range": 0.4997391732434754,
        "drift_min_amp": 0.01644281429758662,
        "drift_amp_range": 0.052719811851424675,
        "powerline_mask_prob": 0.10960157268851577,
        "powerline_min_amp": 0.007957142776487727,
        "powerline_amp_range": 0.033408420749130186,
        "powerline_max_harmonic": 2,
        "channel_mask_prob": 0.10996086493772205,
        "time_num_masks": 8,
        "freq_num_masks": 0,
        "noise_min_snr_db": 27.239461721851817,
        "noise_snr_range_db": 18.431455416940328,
        "noise_apply_prob": 0.6166526983768653,
    },
    {   # Trial #33: val_mae=0.254555
        "gain_min_gain": 0.6704222852108568,
        "gain_range": 0.20819934336868363,
        "gain_mask_prob": 0.15632324013371742,
        "warp_sigma": 0.0817937653676789,
        "warp_num_knots": 5,
        "warp_mask_prob": 0.40869375552770216,
        "drift_mask_prob": 0.33614002836765715,
        "drift_min_freq": 0.047970493750731196,
        "drift_freq_range": 0.3553783525495311,
        "drift_min_amp": 0.009254171940786247,
        "drift_amp_range": 0.015917747781667998,
        "powerline_mask_prob": 0.31568816210015656,
        "powerline_min_amp": 0.019172718321571256,
        "powerline_amp_range": 0.03516945689677829,
        "powerline_max_harmonic": 2,
        "channel_mask_prob": 0.24893248990470065,
        "time_num_masks": 4,
        "freq_num_masks": 19,
        "noise_min_snr_db": 11.415047416332504,
        "noise_snr_range_db": 17.005923564909035,
        "noise_apply_prob": 0.28081903368328135,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_command(
    *,
    trial_number: int,
    gpus: str,
    max_epochs: int,
    trial_dir: str,
    egoemg_memmap_dir: str,
    # ── RandomGain ──
    gain_min_gain: float,
    gain_max_gain: float,
    gain_mask_prob: float,
    # ── RandomMagnitudeWarping ──
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
    # ── RandomChannelMask ──
    channel_mask_prob: float,
    # ── RandomTimeMask ──
    time_num_masks: int,
    # ── RandomFrequencyMask ──
    freq_num_masks: int,
    # ── RandomGaussianNoise ──
    noise_min_snr_db: float,
    noise_max_snr_db: float,
    noise_apply_prob: float,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    """Build the ``python -m emg2pose.train`` command for a single trial."""
    overrides = [
        f"experiment={EXPERIMENT}",
        f"egoemg_memmap_dir={egoemg_memmap_dir}",
        f"trainer.devices=[{gpus}]",
        "+trainer.strategy=ddp",
        f"trainer.max_epochs={max_epochs}",
        f"seed={trial_number}",
        f"hydra.run.dir={trial_dir}",
        # ── RandomGain ──
        f"batch_augmentation.random_gain.min_gain={gain_min_gain:.4f}",
        f"batch_augmentation.random_gain.max_gain={gain_max_gain:.4f}",
        f"batch_augmentation.random_gain.mask_prob={gain_mask_prob:.4f}",
        # ── RandomMagnitudeWarping ──
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
        # ── RandomChannelMask ──
        f"batch_augmentation.channel_mask.mask_prob={channel_mask_prob:.4f}",
        # ── RandomTimeMask ──
        f"batch_augmentation.time_mask.num_masks={time_num_masks}",
        # ── RandomFrequencyMask ──
        f"batch_augmentation.freq_mask.num_masks={freq_num_masks}",
        # ── RandomGaussianNoise ──
        f"batch_augmentation.gaussian_noise.min_snr_db={noise_min_snr_db:.4f}",
        f"batch_augmentation.gaussian_noise.max_snr_db={noise_max_snr_db:.4f}",
        f"batch_augmentation.gaussian_noise.apply_prob={noise_apply_prob:.4f}",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    return [sys.executable, "-m", "emg2pose.train", *overrides]


def parse_val_mae(stdout: str) -> float | None:
    """Extract ``val_mae`` from the *val_metrics* section of pprint output."""
    match = re.search(
        r"'val_metrics':\s*\[.*?'val_mae':\s*([\d.eE+-]+)", stdout, re.DOTALL
    )
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def make_objective(
    gpus: str,
    max_epochs: int,
    egoemg_memmap_dir: str,
    study_name: str,
    extra_overrides: list[str] | None = None,
):
    """Return an Optuna objective function closed over fixed settings."""

    def objective(trial: optuna.Trial) -> float:
        # ── RandomGain ──────────────────────────────────────────────────
        gain_min_gain = trial.suggest_float("gain_min_gain", 0.5, 0.95)
        gain_range = trial.suggest_float("gain_range", 0.02, 0.5)
        gain_max_gain = gain_min_gain + gain_range
        gain_mask_prob = trial.suggest_float("gain_mask_prob", 0.0, 1.0)

        trial.set_user_attr("gain_max_gain", gain_max_gain)

        # ── RandomMagnitudeWarping ──────────────────────────────────────
        warp_sigma = trial.suggest_float("warp_sigma", 0.02, 0.3)
        warp_num_knots = trial.suggest_int("warp_num_knots", 4, 16)
        warp_mask_prob = trial.suggest_float("warp_mask_prob", 0.0, 1.0)

        # ── BaselineDrift ───────────────────────────────────────────────
        drift_mask_prob = trial.suggest_float("drift_mask_prob", 0.0, 0.5)
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

        # ── RandomChannelMask ───────────────────────────────────────────
        channel_mask_prob = trial.suggest_float("channel_mask_prob", 0.0, 0.5)

        # ── RandomTimeMask ──────────────────────────────────────────────
        time_num_masks = trial.suggest_int("time_num_masks", 0, 10)

        # ── RandomFrequencyMask ─────────────────────────────────────────
        freq_num_masks = trial.suggest_int("freq_num_masks", 0, 20)

        # ── RandomGaussianNoise ────────────────────────────────────────
        noise_min_snr_db = trial.suggest_float("noise_min_snr_db", 10.0, 45.0)
        noise_snr_range_db = trial.suggest_float("noise_snr_range_db", 2.0, 25.0)
        noise_max_snr_db = min(noise_min_snr_db + noise_snr_range_db, 50.0)
        noise_apply_prob = trial.suggest_float("noise_apply_prob", 0.1, 1.0)

        trial.set_user_attr("noise_max_snr_db", noise_max_snr_db)

        # ── unique output directory ──────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trial_dir = os.path.join(
            PROJECT_ROOT,
            "logs",
            "optuna_aug",
            study_name,
            f"trial_{trial.number:04d}_{ts}",
        )

        # ── build & run ──────────────────────────────────────────────────
        cmd = build_command(
            trial_number=trial.number,
            gpus=gpus,
            max_epochs=max_epochs,
            trial_dir=trial_dir,
            egoemg_memmap_dir=egoemg_memmap_dir,
            gain_min_gain=gain_min_gain,
            gain_max_gain=gain_max_gain,
            gain_mask_prob=gain_mask_prob,
            warp_sigma=warp_sigma,
            warp_num_knots=warp_num_knots,
            warp_mask_prob=warp_mask_prob,
            drift_mask_prob=drift_mask_prob,
            drift_min_freq=drift_min_freq,
            drift_max_freq=drift_max_freq,
            drift_min_amp=drift_min_amp,
            drift_max_amp=drift_max_amp,
            powerline_mask_prob=powerline_mask_prob,
            powerline_min_amp=powerline_min_amp,
            powerline_max_amp=powerline_max_amp,
            powerline_max_harmonic=powerline_max_harmonic,
            channel_mask_prob=channel_mask_prob,
            time_num_masks=time_num_masks,
            freq_num_masks=freq_num_masks,
            noise_min_snr_db=noise_min_snr_db,
            noise_max_snr_db=noise_max_snr_db,
            noise_apply_prob=noise_apply_prob,
            extra_overrides=extra_overrides,
        )

        log.info("Trial %d — cmd: %s", trial.number, " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        # ── handle failures ──────────────────────────────────────────────
        if result.returncode != 0:
            log.error(
                "Trial %d FAILED (rc=%d).  stderr tail:\n%s",
                trial.number,
                result.returncode,
                textwrap.indent(
                    "\n".join(result.stderr.strip().splitlines()[-30:]),
                    "    ",
                ),
            )
            return float("inf")

        # ── parse result ─────────────────────────────────────────────────
        val_mae = parse_val_mae(result.stdout)
        if val_mae is None:
            log.error(
                "Trial %d — could not parse val_mae from stdout.  stdout tail:\n%s",
                trial.number,
                textwrap.indent(
                    "\n".join(result.stdout.strip().splitlines()[-20:]),
                    "    ",
                ),
            )
            return float("inf")

        log.info("Trial %d — val_mae = %.6f", trial.number, val_mae)
        return val_mae

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna full augmentation search for EgoEMG middle model (GPU batch, DDP)"
    )
    p.add_argument(
        "--gpus",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated GPU indices for DDP (default: 1,2,3,4,5)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=60,
        help="Number of Optuna trials (default: 60)",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=150,
        help="Max training epochs per trial (default: 150)",
    )
    p.add_argument(
        "--storage",
        default=DEFAULT_STORAGE,
        help=f"Optuna storage URL (default: {DEFAULT_STORAGE})",
    )
    p.add_argument(
        "--study-name",
        default="egoemg-aug-v5",
        help="Optuna study name for resume (default: egoemg-aug-v5)",
    )
    p.add_argument(
        "--egoemg-memmap-dir",
        default="/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap",
        help="Path to EgoEMG memmap directory",
    )
    p.add_argument(
        "--extra-override",
        action="append",
        dest="extra_overrides",
        default=[],
        help="Extra Hydra overrides to pass to every trial (repeatable)",
    )
    p.add_argument(
        "--sampler",
        choices=["tpe", "random"],
        default="tpe",
        help="Optuna sampler (default: tpe)",
    )
    p.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default="minimize",
        help="Optimization direction (default: minimize)",
    )
    p.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help="Random seed for the Optuna sampler (default: 42)",
    )
    p.add_argument(
        "--no-enqueue",
        action="store_true",
        default=False,
        help="Skip enqueuing v3 best params",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── sampler ──────────────────────────────────────────────────────────
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.sampler_seed)

    # ── study ────────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        sampler=sampler,
        load_if_exists=True,
    )

    log.info(
        "Study %r — %d completed trials already in storage.",
        args.study_name,
        len(study.trials),
    )

    # ── inherit v3 trial history for TPE prior ────────────────────────────
    if len(study.trials) == 0:
        # Copy all completed v3 trials into this study so TPE learns from them
        try:
            v3 = optuna.load_study(
                study_name="egoemg-aug-v3",
                storage=args.storage,
            )
            v3_completed = [
                t for t in v3.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            for t in v3_completed:
                study.add_trial(
                    optuna.trial.create_trial(
                        params=t.params,
                        distributions=t.distributions,
                        value=t.value,
                    )
                )
            log.info("Copied %d completed v3 trials into study.", len(v3_completed))
        except Exception:
            log.warning("Could not copy v3 trials — continuing without prior.", exc_info=True)

        # Enqueue top v3 params as first trials to evaluate
        for i, params in enumerate(V3_TOP_TRIALS):
            study.enqueue_trial(params)
        log.info("Enqueued %d top v3 trials as warm start.", len(V3_TOP_TRIALS))

    objective = make_objective(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        egoemg_memmap_dir=args.egoemg_memmap_dir,
        study_name=args.study_name,
        extra_overrides=args.extra_overrides or None,
    )

    study.optimize(objective, n_trials=args.n_trials)

    # ── report ───────────────────────────────────────────────────────────
    best = study.best_trial
    log.info("===== Best trial: %d =====", best.number)
    log.info("  val_mae = %.6f", best.value)
    log.info("  params:")
    for k, v in best.params.items():
        log.info("    %s = %s", k, v)
    log.info("  user_attrs:")
    for k, v in best.user_attrs.items():
        log.info("    %s = %s", k, v)


if __name__ == "__main__":
    main()
