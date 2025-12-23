# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2pose import transforms
from emg2pose.lightning_vqvae import JointAngleVQVAEModule
from emg2pose.transforms import Transform


log = logging.getLogger(__name__)


def make_data_module(config: DictConfig):
    def _full_paths(root: str, dataset: ListConfig) -> list[Path]:
        sessions = dataset
        return [
            Path(root).expanduser().joinpath(f"{session}.hdf5") for session in sessions
        ]

    splits = instantiate(config.data_split)
    train_sessions = _full_paths(config.data_location, splits["train"])
    val_sessions = _full_paths(config.data_location, splits["val"])
    test_sessions = _full_paths(config.data_location, splits["test"])

    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        test_sessions=test_sessions,
    )

    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    datamodule.train_transforms = _build_transform(config.transforms.train)
    datamodule.val_transforms = _build_transform(config.transforms.val)
    datamodule.test_transforms = _build_transform(config.transforms.test)

    return datamodule


def make_lightning_module(config: DictConfig):
    return JointAngleVQVAEModule(
        vqvae_conf=config.vqvae,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        repr_conf=config.angle_representation,
    )


def train(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    pl.seed_everything(config.seed, workers=True)
    matmul_precision = config.get("matmul_precision")
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))

    if config.checkpoint is not None:
        log.info(f"Loading from checkpoint {config.checkpoint}")
        module = JointAngleVQVAEModule.load_from_checkpoint(
            config.checkpoint,
            vqvae_conf=config.vqvae,
            optimizer_conf=config.optimizer,
            lr_scheduler_conf=config.lr_scheduler,
        )
    else:
        log.info("Instantiating VQ-VAE LightningModule")
        module = make_lightning_module(config)

    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = make_data_module(config)

    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    trainer = pl.Trainer(**config.trainer, callbacks=callbacks)

    results = {}
    if config.train:
        trainer.fit(module, datamodule)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is None:
            raise RuntimeError("No checkpoint callback found in trainer")
        best_checkpoint_path = checkpoint_callback.best_model_path
        module = module.__class__.load_from_checkpoint(best_checkpoint_path)
        results["best_checkpoint"] = best_checkpoint_path

    if config.eval:
        module.eval()
        val_metrics = trainer.validate(module, datamodule)
        test_metrics = trainer.test(module, datamodule)
        results["val_metrics"] = val_metrics
        results["test_metrics"] = test_metrics

    pprint.pprint(results, sort_dicts=False)
    return results


@hydra.main(config_path="../config", config_name="vqvae_base", version_base="1.1")
def cli(config: DictConfig):
    train(config)


if __name__ == "__main__":
    cli()
