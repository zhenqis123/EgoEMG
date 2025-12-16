# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

from collections.abc import Mapping, Sequence
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from emg2pose import utils
from emg2pose.datasets.emg2pose_dataset import WindowedEmgDataset
from emg2pose.metrics import get_default_metrics
from emg2pose.models.modules import BaseModule
from hydra.utils import instantiate

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        stride: int | None,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        val_test_window_length: int | None = None,
        val_test_stride: int | None = None,
        skip_ik_failures: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.stride = stride
        self.val_test_stride = val_test_stride if val_test_stride is not None else stride
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.skip_ik_failures = skip_ik_failures

    def setup(self, stage: str | None = None) -> None:
        # train
        self.train_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.train_transforms,
                window_length=self.window_length,
                stride=self.stride,
                padding=self.padding,
                jitter=True,
                skip_ik_failures=self.skip_ik_failures,
            )
            for hdf5_path in tqdm(self.train_sessions, desc="Building train datasets")
        ])

        # val
        self.val_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.val_transforms,
                window_length=self.val_test_window_length,
                stride=self.val_test_stride,
                padding=self.padding,
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
            )
            for hdf5_path in tqdm(self.val_sessions, desc="Building val datasets")
        ])

        # test
        self.test_dataset = ConcatDataset([
            WindowedEmgDataset(
                hdf5_path,
                transform=self.test_transforms,
                window_length=self.val_test_window_length,
                stride=self.val_test_stride,
                padding=(0, 0),
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
            )
            for hdf5_path in tqdm(self.test_sessions, desc="Building test datasets")
        ])


    def train_dataloader(self) -> DataLoader:
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs = dict(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)

    def test_dataloader(self) -> DataLoader:
        kwargs = dict(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)

class EmgPredictionModule(pl.LightningModule):
    def __init__(
        self,
        module_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        loss_weights: dict[str, float] | None = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.model: BaseModule = instantiate(module_conf, _convert_="all")
        self.provide_initial_pos = bool(getattr(self.model, "provide_initial_pos", False))
        self.loss_weights = loss_weights or {"mae": 1}
        self._warned_emg_nan = False

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
        preds, targets, mask = self.forward(batch)

        # Build final loss/metric mask
        valid_mask = mask.bool()
        # Compute metrics
        metrics = {}
        for metric in self.metrics_list:
            metrics.update(metric(preds, targets, valid_mask, stage))
        self.log_dict(metrics, sync_dist=True)

        # Compute loss
        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            loss += metrics[f"{stage}_{loss_name}"] * weight
        self.log(f"{stage}_loss", loss, sync_dist=True)

        return loss
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        result = self._step(batch, stage="train")
        return result

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

    def build_valid_mask(
        self,
        base_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Combine base IK mask and finite-value check."""
        mask = base_mask.bool()

        # Drop any timestep containing NaN/Inf in targets across joints
        finite = torch.isfinite(targets).all(dim=1)
        mask = mask & finite

        # Warn if everything is masked to avoid empty tensors in losses
        if mask.sum() == 0:
            log.warning("All samples masked out after combining IK/interp/finite checks.")

        return mask
