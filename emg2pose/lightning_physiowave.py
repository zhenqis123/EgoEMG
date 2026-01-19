import logging

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

from emg2pose import utils
from emg2pose.metrics import get_default_metrics

log = logging.getLogger(__name__)


class PhysioWaveTemporalLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_conf,
        optimizer_conf,
        lr_scheduler_conf,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(model_conf, _convert_="all")
        self.loss_weights = loss_weights or {"mae": 1.0}
        self.regression_metrics = get_default_metrics()
        self.regression_mode = getattr(self.model, "regression_mode", "temporal")

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        return self.model(emg)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            list(self.parameters()),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )

    def _step(self, batch, stage: str) -> torch.Tensor:
        emg = batch["emg"]
        targets = batch["joint_angles"]
        mask = batch["label_valid_mask"]
        preds = self.model(emg)
        if self.regression_mode == "pooled":
            preds = preds.unsqueeze(-1)
            targets = targets[..., -1:].contiguous()
            mask = mask[..., -1:].contiguous()
        valid_mask = self.build_valid_mask(mask, targets)

        metrics = {}
        for metric in self.regression_metrics:
            metrics.update(metric(preds, targets, valid_mask, stage))
        self.log_dict(metrics, sync_dist=True, batch_size=emg.shape[0])

        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            loss += metrics.get(f"{stage}_{loss_name}", 0.0) * weight
        self.log(f"{stage}_loss", loss, sync_dist=True, batch_size=emg.shape[0])
        return loss

    def build_valid_mask(
        self,
        base_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        mask = base_mask.bool()
        finite = torch.isfinite(targets).all(dim=1)
        mask = mask & finite
        if mask.sum() == 0:
            log.warning("All samples masked out after combining IK/interp/finite checks.")
        return mask
