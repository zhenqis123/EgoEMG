# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Mapping

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import functional as F

from emg2pose import utils
from emg2pose.metrics import get_default_metrics

log = logging.getLogger(__name__)


class EmgConformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(model_conf, _convert_="all")
        self.loss_weights = loss_weights or {"mae": 1}
        self.regression_metrics = get_default_metrics()

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = self.model(batch["emg"])
        joint_angles = batch["joint_angles"]
        mask = batch["label_valid_mask"]
        start = self.model.left_context
        stop = None if self.model.right_context == 0 else -self.model.right_context
        targets = joint_angles[..., slice(start, stop)]
        mask = mask[..., slice(start, stop)]
        n_time = targets.shape[-1]
        preds = self._align_predictions(preds, n_time)
        mask = self._align_mask(mask, n_time)
        return preds, targets, mask

    def _align_predictions(self, preds: torch.Tensor, n_time: int) -> torch.Tensor:
        if preds.shape[-1] == n_time:
            return preds
        return F.interpolate(preds, size=n_time, mode="linear")

    def _align_mask(self, mask: torch.Tensor, n_time: int) -> torch.Tensor:
        if mask.shape[-1] == n_time:
            return mask
        mask = mask[:, None].to(torch.float32)
        aligned = F.interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        if (
            getattr(self.hparams, "datamodule", None)
            and self.hparams.datamodule.get("norm_mode") == "batch"
        ):
            emg = batch["emg"]
            mean = emg.mean()
            std = emg.std()
            batch["emg"] = (emg - mean) / (std + 1e-6)
        preds, targets, mask = self.forward(batch)
        valid_mask = mask.bool()
        batch_size = batch["emg"].shape[0]

        metrics = {}
        for metric in self.regression_metrics:
            metrics.update(metric(preds, targets, valid_mask, stage))
        self.log_dict(metrics, sync_dist=True, batch_size=batch_size)

        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            loss += metrics.get(f"{stage}_{loss_name}", 0.0) * weight
        self.log(f"{stage}_loss", loss, sync_dist=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )
