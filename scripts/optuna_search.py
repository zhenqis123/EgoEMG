#!/usr/bin/env python

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import optuna
from hydra import compose, initialize
from omegaconf import OmegaConf

from emg2pose.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna search over batch_size and learning rate using val_mae."
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="base",
        help="Hydra config name (without extension), default 'base'.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Hydra config directory containing config files.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help='Optuna storage URL, e.g., "sqlite:///optuna.db". If None, runs in-memory.',
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="emg2pose_search",
        help="Name of the Optuna study.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Candidate batch sizes.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-4,
        help="Minimum learning rate (log-uniform).",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=3e-3,
        help="Maximum learning rate (log-uniform).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs to shorten trials (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with initialize(config_path=str(args.config_dir), version_base=None):
        base_cfg = compose(config_name=args.config_name)

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)

        # Suggest hyperparameters
        cfg.batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)
        cfg.optimizer.lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)

        # Optionally shorten training per trial
        if args.max_epochs is not None:
            cfg.trainer.max_epochs = args.max_epochs
            # Speed up early stopping if present
            callbacks = cfg.get("callbacks", [])
            for cb in callbacks:
                if cb.get("_target_") == "pytorch_lightning.callbacks.EarlyStopping":
                    cb["patience"] = min(cb.get("patience", 50), max(5, args.max_epochs // 2))
            cfg.callbacks = callbacks

        # Run train/validate
        results = train(cfg)
        if "val_metrics" not in results or not results["val_metrics"]:
            return float("inf")

        val_metrics = results["val_metrics"][0]
        # Prefer val_mae; fallback to val_loss if missing
        score = val_metrics.get("val_mae", val_metrics.get("val_loss"))
        if score is None:
            return float("inf")
        return float(score)

    study = optuna.create_study(
        direction="minimize",
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective, n_trials=args.trials)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
