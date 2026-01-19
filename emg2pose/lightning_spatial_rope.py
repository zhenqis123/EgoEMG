from __future__ import annotations

import logging
from collections.abc import Mapping

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from emg2pose import utils
from emg2pose.metrics import get_default_metrics

log = logging.getLogger(__name__)


class SpatialRoPELightningModule(pl.LightningModule):
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
        if preds.ndim == 2:
            preds = preds[..., None]
        pred_t = preds.shape[-1]
        target_t = targets.shape[-1]
        if pred_t != target_t:
            patch_size = getattr(self.model.featurizer, "patch_size", None)
            if patch_size is None:
                preds = self.model.align_predictions(preds, target_t)
                mask = self.model.align_mask(mask, target_t)
                return preds, targets, mask
            indices = torch.arange(
                pred_t, device=targets.device, dtype=torch.long
            ) * int(patch_size)
            if indices[-1] >= target_t:
                indices = indices[indices < target_t]
                preds = preds[..., : indices.numel()]
            targets = targets.index_select(-1, indices)
            mask = mask.index_select(-1, indices)
        return preds, targets, mask

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
        batch_size = batch["emg"].shape[0]
        valid_mask = mask.bool()

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
