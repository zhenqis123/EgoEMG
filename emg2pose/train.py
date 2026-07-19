# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import os
import pprint
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import hydra
import pytorch_lightning as pl

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    category=UserWarning,
)

from emg2pose.datamodule import make_data_module
from emg2pose.lightning import EmgPredictionModule
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch


log = logging.getLogger(__name__)


def _ensure_model_summary_callback(
    callbacks: list[pl.Callback], config: DictConfig
) -> list[pl.Callback]:
    """Ensure a ModelSummary callback is present unless explicitly disabled."""
    wants_summary = bool(config.get("default_model_summary", True))
    if not wants_summary:
        return callbacks

    if any(isinstance(cb, pl.callbacks.ModelSummary) for cb in callbacks):
        return callbacks

    max_depth = int(config.get("model_summary_max_depth", 2))
    return [pl.callbacks.ModelSummary(max_depth=max_depth), *callbacks]


def make_lightning_module(config: DictConfig):
    """Create lightning module from experiment config."""
    module_target = str(config.module.get("_target_", ""))
    if module_target == "emg2pose.models.modules.emgformer_pretrain.EmgformerPretrain":
        raise ValueError(
            "This experiment config targets EmgformerPretrain and must be run via "
            "`python -m emg2pose.train_pretrain ...`, not `python -m emg2pose.train ...`."
        )
    return EmgPredictionModule(
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        loss_weights=config.loss_weights,
        pretrained_checkpoint=config.get("pretrained_checkpoint"),
        pretrained_strict=config.get("pretrained_strict", False),
        freeze_backbone=config.get("freeze_backbone", False),
        pretrained_emg_checkpoint=config.get("pretrained_emg_checkpoint"),
        stage2_vision_checkpoint=config.get("stage2_vision_checkpoint"),
        component_lr_scales=config.get("component_lr_scales"),
        ignore_head_tail_dims=config.get("ignore_head_tail_dims", 0),
        datamodule=config.datamodule,
        batch_augmentation=config.get("batch_augmentation"),
        val_episode_name_mapping=config.get("val_episode_name_mapping"),
    )

def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor] | None:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(
            checkpoint["model_state_dict"], dict
        ):
            return checkpoint["model_state_dict"]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint  # raw state dict
    return None


def _is_pretrain_checkpoint(state_dict: dict[str, torch.Tensor] | None) -> bool:
    if not state_dict:
        return False
    pretrain_prefixes = (
        "model.recon_head.",
        "model.gesture_head.",
        "model.keystroke_head.",
        "model.angle_head.",  # Pretrain uses angle_head, regular uses head
        "model.mask_embedding",
        "model.projection.",
    )
    return any(key.startswith(pretrain_prefixes) for key in state_dict.keys())


def train(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
):
    # import torch
    # torch.autograd.set_detect_anomaly(True)
    
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    matmul_precision = config.get("matmul_precision")
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))

    if config.checkpoint is not None:
        log.info(f"Loading from checkpoint {config.checkpoint}")
        ckpt_path = Path(config.checkpoint).expanduser()
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        strict = bool(config.get("pretrained_strict", False))

        if _is_pretrain_checkpoint(state_dict):
            log.info("Detected pretrain checkpoint. Loading backbone and angle head.")
        else:
            log.info("Detected regular checkpoint. Loading backbone and matching head weights.")

        module = make_lightning_module(config)
        module._load_pretrained_backbone(str(ckpt_path), strict=strict)
        module._load_pretrained_angle_head(str(ckpt_path), strict=strict)
    else:
        log.info(f"Instantiating LightningModule {EmgPredictionModule}")
        module = make_lightning_module(config)

    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = make_data_module(config)

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]
    callbacks = _ensure_model_summary_callback(callbacks, config)

    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    logger = None
    if "logger" in config:
        logger = instantiate(config.logger)

    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    results = {}
    if config.train:

        # Train
        trainer.fit(module, datamodule)

        # Load the best checkpoint (skip if no val data to select best)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is None:
            raise RuntimeError("No checkpoint callback found in trainer")
        best_checkpoint_path = checkpoint_callback.best_model_path
        if best_checkpoint_path and os.path.isfile(best_checkpoint_path):
            module = module.__class__.load_from_checkpoint(best_checkpoint_path)
        else:
            best_checkpoint_path = checkpoint_callback.last_model_path
            if best_checkpoint_path and os.path.isfile(best_checkpoint_path):
                module = module.__class__.load_from_checkpoint(best_checkpoint_path)

        results["best_checkpoint"] = best_checkpoint_path

    if config.eval:

        # Compute validation and test set metrics
        module.eval()
        val_metrics = trainer.validate(module, datamodule)
        test_metrics = trainer.test(module, datamodule)

        results["val_metrics"] = val_metrics
        results["test_metrics"] = test_metrics

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


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def cli(config: DictConfig):
    train(config)


if __name__ == "__main__":
    cli()
