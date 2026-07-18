# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint
from collections.abc import Callable, Sequence

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from emg2pose.datamodule import make_data_module
from emg2pose.lightning_pretrain import EmgPretrainModule

log = logging.getLogger(__name__)


def make_lightning_module(config: DictConfig):
    return EmgPretrainModule(
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        gesture_spaces=config.gesture_spaces,
        loss_weights=config.loss_weights,
        mask_conf=config.masking,
        recon_loss=config.get("recon_loss", "mse"),
        angle_loss=config.get("angle_loss", "mae"),
        label_smoothing=config.get("label_smoothing", 0.0),
        datamodule=config.datamodule,
    )


def train(config: DictConfig, extra_callbacks: Sequence[Callable] | None = None):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")
    pl.seed_everything(config.seed, workers=True)

    matmul_precision = config.get("matmul_precision")
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))

    if config.checkpoint is not None:
        log.info(f"Loading from checkpoint {config.checkpoint}")
        module = EmgPretrainModule.load_from_checkpoint(
            config.checkpoint,
            module_conf=config.module,
            optimizer_conf=config.optimizer,
            lr_scheduler_conf=config.lr_scheduler,
            gesture_spaces=config.gesture_spaces,
            loss_weights=config.loss_weights,
            mask_conf=config.masking,
            recon_loss=config.get("recon_loss", "mse"),
            angle_loss=config.get("angle_loss", "mae"),
            label_smoothing=config.get("label_smoothing", 0.0),
            datamodule=config.datamodule,
        )
    else:
        module = make_lightning_module(config)

    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = make_data_module(config)

    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    logger = None
    if "logger" in config:
        logger = instantiate(config.logger)

    trainer_cfg = OmegaConf.to_container(config.trainer, resolve=True)
    if not torch.cuda.is_available():
        trainer_cfg["accelerator"] = "cpu"
        trainer_cfg["devices"] = 1
        trainer_cfg["precision"] = 32
    trainer = pl.Trainer(**trainer_cfg, callbacks=callbacks, logger=logger)

    results = {}
    if config.train:
        trainer.fit(module, datamodule)
        if trainer.checkpoint_callback is not None:
            results["last_checkpoint"] = trainer.checkpoint_callback.best_model_path

    if config.eval:
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


@hydra.main(config_path="../config", config_name="pretrain", version_base="1.1")
def cli(config: DictConfig):
    train(config)


if __name__ == "__main__":
    cli()
