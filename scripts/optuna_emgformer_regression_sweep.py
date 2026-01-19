#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from typing import Sequence

import optuna
import pytorch_lightning as pl
from torch.multiprocessing.spawn import ProcessRaisedException
from hydra import compose, initialize
from omegaconf import DictConfig, open_dict
from emg2pose.train import train as train_fn


class _OptunaPruningCallback(pl.callbacks.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str) -> None:
        super().__init__()
        self._trial = trial
        self._monitor = monitor

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        metrics = trainer.callback_metrics
        if self._monitor not in metrics:
            return
        value = metrics[self._monitor]
        try:
            score = float(value)
        except (TypeError, ValueError):
            return
        step = trainer.current_epoch
        self._trial.report(score, step)
        if self._trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at epoch {step} with {self._monitor}={score}."
            )


def _parse_devices(devices: str) -> list[int]:
    return [int(item) for item in devices.split(",") if item.strip() != ""]


def _build_base_config(args: argparse.Namespace) -> DictConfig:
    overrides = [
        "experiment=emgformer/regression_emgformer_small",
        f"data_split={args.data_split}",
        f"trainer.max_epochs={args.max_epochs}",
        "train=True",
        "eval=True",
    ]
    if args.data_location:
        overrides.append(f"data_location={args.data_location}")
    if args.batch_size is not None:
        overrides.append(f"batch_size={args.batch_size}")

    with initialize(version_base="1.1", config_path="../config"):
        return compose(config_name="base", overrides=overrides)


def _apply_trial_params(cfg: DictConfig, trial: optuna.Trial) -> None:
    cfg.module.decoder.dropout = trial.suggest_float(
        "module.decoder.dropout", 0.05, 0.25
    )

    cfg.datamodule.window_length = 7790
    cfg.datamodule.val_test_window_length = 7790

    cfg.transforms.train[3].mask_prob = trial.suggest_float(
        "transforms.train.3.mask_prob", 0.2, 0.7
    )

    cfg.transforms.train[4].num_masks = trial.suggest_categorical(
        "transforms.train.4.num_masks", [0, 3, 5, 7]
    )

    cfg.transforms.train[5].num_masks = trial.suggest_categorical(
        "transforms.train.5.num_masks", [0, 3, 5, 7]
    )

    cfg.transforms.train[6].apply_prob = trial.suggest_float(
        "transforms.train.6.apply_prob", 0.2, 0.7
    )


def _set_trainer_runtime(
    cfg: DictConfig, devices: Sequence[int], strategy: str | None, seed: int
) -> None:
    cfg.trainer.devices = list(devices)
    cfg.seed = seed
    if strategy and len(devices) > 1:
        if strategy == "ddp":
            raise ValueError(
                "Optuna sweeps cannot use 'ddp' because Lightning re-launches the "
                "script per rank. Use 'ddp_spawn' or 'ddp_fork' instead."
            )
        with open_dict(cfg.trainer):
            cfg.trainer.strategy = strategy


def _is_pruning_exception(exc: BaseException) -> bool:
    if isinstance(exc, optuna.TrialPruned):
        return True
    return "optuna.exceptions.TrialPruned" in str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--strategy", type=str, default="ddp_spawn")
    parser.add_argument("--study-name", type=str, default="emgformer_regression_small")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--data-split", type=str, default="mini_split")
    parser.add_argument("--data-location", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    devices = _parse_devices(args.devices)
    base_cfg = _build_base_config(args)

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        _apply_trial_params(cfg, trial)
        _set_trainer_runtime(cfg, devices, args.strategy, args.seed)

        pruning_cb = _OptunaPruningCallback(trial, monitor=cfg.monitor_metric)
        try:
            results = train_fn(cfg, extra_callbacks=[pruning_cb])
        except ProcessRaisedException as exc:
            # ddp_spawn wraps child exceptions; map pruning to a proper Optuna signal.
            if _is_pruning_exception(exc):
                raise optuna.TrialPruned(str(exc))
            raise
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            if _is_pruning_exception(exc):
                raise optuna.TrialPruned(str(exc))
            raise
        if not cfg.eval or "val_metrics" not in results:
            raise RuntimeError("eval=True is required to report validation metrics.")
        val_metrics = results["val_metrics"][0]
        if cfg.monitor_metric not in val_metrics:
            raise KeyError(
                f"Monitor metric '{cfg.monitor_metric}' missing from val metrics."
            )
        return float(val_metrics[cfg.monitor_metric])

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    storage = None
    if args.storage:
        storage = optuna.storages.RDBStorage(args.storage)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage if args.storage else None,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True if args.storage else False,
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
