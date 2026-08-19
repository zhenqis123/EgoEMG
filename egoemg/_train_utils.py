"""Shared training infrastructure for the supervised and pretrain entrypoints.

Both `egoemg/train.py` and `egoemg/train_pretrain.py` need the same setup:
seed/matmul, callbacks, logger, trainer construction (with CPU fallback),
fit/eval/optuna-return. Extracted here to avoid ~150 lines of duplication and
to ensure both entrypoints share the same CPU-fallback safety net.
"""
from __future__ import annotations

import logging
import os
import pprint
from collections.abc import Callable, Sequence
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def setup_runtime(config: DictConfig) -> None:
    """Seed determinism + matmul precision. Call once at the start of train()."""
    pl.seed_everything(config.seed, workers=True)
    matmul_precision = config.get("matmul_precision")
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))


def build_callbacks(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
    *,
    ensure_model_summary: bool = False,
) -> list[pl.Callback]:
    """Instantiate callback configs + merge extras + optionally prepend ModelSummary."""
    callback_configs = config.get("callbacks", [])
    callbacks: list[pl.Callback] = [instantiate(cfg) for cfg in callback_configs]
    if ensure_model_summary:
        callbacks = _ensure_model_summary_callback(callbacks, config)
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)
    return callbacks


def _ensure_model_summary_callback(
    callbacks: list[pl.Callback], config: DictConfig
) -> list[pl.Callback]:
    """Prepend a ModelSummary callback if enabled and not already present."""
    if not config.get("default_model_summary", True):
        return callbacks
    if any(isinstance(cb, pl.callbacks.ModelSummary) for cb in callbacks):
        return callbacks
    max_depth = config.get("model_summary_max_depth", 2)
    return [pl.callbacks.ModelSummary(max_depth=max_depth), *callbacks]


def build_logger(config: DictConfig) -> Any:
    """Instantiate the logger if configured, else None."""
    if "logger" in config:
        return instantiate(config.logger)
    return None


def build_trainer(
    config: DictConfig,
    callbacks: list[pl.Callback],
    logger: Any,
) -> pl.Trainer:
    """Construct pl.Trainer from config, with a CPU fallback when CUDA is absent.

    Both entrypoints now share this fallback (previously only train_pretrain had it,
    so the supervised entrypoint would crash on CPU-only machines).
    """
    trainer_cfg = OmegaConf.to_container(config.trainer, resolve=True)
    if not torch.cuda.is_available():
        trainer_cfg["accelerator"] = "cpu"
        trainer_cfg["devices"] = 1
        trainer_cfg["precision"] = 32
    return pl.Trainer(**trainer_cfg, callbacks=callbacks, logger=logger)


def run_train_eval(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    datamodule,
    config: DictConfig,
    *,
    reload_best: bool = True,
) -> dict:
    """Run fit + eval + return results, with Optuna MULTIRUN support.

    Shared by both entrypoints. When `reload_best=True` (supervised default),
    reloads the best checkpoint into `module` after fit for clean eval.
    Pretrain sets `reload_best=False` (keeps last).

    Returns the results dict, or a float (monitor_metric) under Optuna MULTIRUN.
    """
    results: dict[str, Any] = {}
    if config.train:
        resume_ckpt = config.get("resume_ckpt", None)
        if resume_ckpt:
            resume_ckpt = os.path.expanduser(resume_ckpt)
            if not os.path.isfile(resume_ckpt):
                raise FileNotFoundError(f"resume_ckpt not found: {resume_ckpt}")
            log.info("Resuming training from checkpoint: %s", resume_ckpt)
        trainer.fit(module, datamodule, ckpt_path=resume_ckpt)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is None:
            raise RuntimeError("No checkpoint callback found in trainer")

        best_checkpoint_path = checkpoint_callback.best_model_path
        if best_checkpoint_path and os.path.isfile(best_checkpoint_path):
            chosen = best_checkpoint_path
            results_key = "best_checkpoint"
        else:
            chosen = checkpoint_callback.last_model_path
            results_key = "last_checkpoint"

        if reload_best and chosen and os.path.isfile(chosen):
            # Our own checkpoint is a trusted local artifact.  It records
            # OmegaConf hyperparameters, which PyTorch 2.6+'s weights-only
            # default cannot deserialize.
            module = module.__class__.load_from_checkpoint(
                chosen, weights_only=False
            )
        if chosen:
            results[results_key] = chosen

    if config.eval:
        module.eval()
        results["val_metrics"] = trainer.validate(module, datamodule)
        results["test_metrics"] = trainer.test(module, datamodule)

    pprint.pprint(results, sort_dicts=False)

    if HydraConfig.initialized() and HydraConfig.get().mode.name == "MULTIRUN":
        if not config.eval or not results.get("val_metrics"):
            raise RuntimeError("Optuna sweeps require eval=True with val metrics.")
        val_metrics = results["val_metrics"][0]
        if config.monitor_metric not in val_metrics:
            raise KeyError(
                f"Monitor metric '{config.monitor_metric}' missing from val metrics."
            )
        return float(val_metrics[config.monitor_metric])

    return results
