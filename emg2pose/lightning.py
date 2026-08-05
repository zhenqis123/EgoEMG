# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytorch_lightning as pl
import torch

from emg2pose import utils
from emg2pose.data import WindowedEmgDataset
from emg2pose.metrics import get_default_metrics
from emg2pose.pose_modules import BasePoseModule
from hydra.utils import instantiate

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader


log = logging.getLogger(__name__)


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        val_test_window_length: int | None = None,
        skip_ik_failures: bool = False,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.skip_ik_failures = skip_ik_failures

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEmgDataset(
                    hdf5_path,
                    transform=self.train_transforms,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                    skip_ik_failures=self.skip_ik_failures,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEmgDataset(
                    hdf5_path,
                    transform=self.val_transforms,
                    window_length=self.val_test_window_length,
                    padding=self.padding,
                    jitter=False,
                    skip_ik_failures=self.skip_ik_failures,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEmgDataset(
                    hdf5_path,
                    transform=self.test_transforms,
                    window_length=self.val_test_window_length,
                    padding=(0, 0),
                    jitter=False,
                    skip_ik_failures=self.skip_ik_failures,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )


class Emg2PoseModule(pl.LightningModule):
    def __init__(
        self,
        network_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        provide_initial_pos: bool = False,
        loss_weights: dict[str, float] | None = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.model: BasePoseModule = instantiate(network_conf, _convert_="all")
        self.provide_initial_pos = provide_initial_pos
        self.loss_weights = loss_weights or {"mae": 1}

        # TODO: add metrics to Hydra config instead
        self.metrics_list = get_default_metrics()

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward(batch, self.provide_initial_pos)

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str = "train"
    ) -> torch.Tensor:

        # Generate predictions
        batch["no_ik_failure"] = self.update_ik_failure_mask(batch["no_ik_failure"])
        preds, targets, no_ik_failure = self.forward(batch)

        # Compute metrics
        metrics = {}
        for metric in self.metrics_list:
            metrics.update(metric(preds, targets, no_ik_failure, stage))
        self.log_dict(metrics, sync_dist=True)

        # Compute loss
        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            loss += metrics[f"{stage}_{loss_name}"] * weight
        self.log(f"{stage}_loss", loss, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(
        self, batch, batch_idx, dataloader_idx: int | None = None
    ) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )

    def update_ik_failure_mask(self, no_ik_failure: torch.Tensor) -> torch.Tensor:
        """Update the mask to only include samples where there are no ik failures."""

        # Mask out samples where the initial position is zero, because state
        # initialization doesn't work under these conditions. Note that the initial
        # position is the left_context'th sample, not the 0th sample.
        mask = no_ik_failure.clone()

        if self.provide_initial_pos:
            mask[~mask[:, self.model.left_context]] = False

        if mask.sum() == 0:
            log.warning("All samples masked out due to missing initial state!")

        return mask
