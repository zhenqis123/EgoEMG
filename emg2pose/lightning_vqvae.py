from __future__ import annotations

import logging
from collections.abc import Mapping

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from emg2pose.models.quantizers.vq import VectorQuantizer
from emg2pose.utils import instantiate_optimizer_and_scheduler
from emg2pose.geometry import (
    angles_to_axis_angle,
    axis_angle_to_angles,
    axis_angle_to_rot6d,
    get_joint_rotation_axes,
    rot6d_to_axis_angle,
)
from emg2pose.metrics import get_default_metrics
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class JointAngleVQVAEModule(pl.LightningModule):
    def __init__(
        self,
        vqvae_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig | None,
        repr_conf: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.repr_mode = str(repr_conf.get("mode", "angle"))
        self.num_joints = int(repr_conf.get("num_joints", 20))
        self.repr_dim = self._get_representation_dim()
        if self.repr_mode != "angle":
            axes = get_joint_rotation_axes(self.num_joints)
            self.register_buffer("joint_axes", axes, persistent=False)

        configured_dim = vqvae_conf.get("input_dim")
        if configured_dim is not None and int(configured_dim) != self.repr_dim:
            log.warning(
                "Overriding vqvae.input_dim=%s with representation dim=%s",
                configured_dim,
                self.repr_dim,
            )
        # Allow adding output_dim in case config struct is locked
        try:
            OmegaConf.set_struct(vqvae_conf, False)
        except Exception:
            pass
        # vqvae_conf = OmegaConf.merge(
        #     vqvae_conf, {"input_dim": self.repr_dim, "output_dim": self.repr_dim}
        # )
        self.model = instantiate(vqvae_conf, _convert_="all")
        self.metrics = get_default_metrics()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _flatten_joint_angles(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        joint_angles = batch["joint_angles"]  # B, C, T
        joint_angles = joint_angles.transpose(1, 2)  # B, T, C
        flat = joint_angles.reshape(-1, joint_angles.shape[-1])
        mask = batch.get("label_valid_mask")
        if mask is None:
            return flat
        mask_flat = mask.reshape(-1)
        return flat[mask_flat]

    def _get_representation_dim(self) -> int:
        if self.repr_mode == "angle":
            return self.num_joints
        if self.repr_mode == "axis_angle":
            return self.num_joints * 3
        if self.repr_mode == "rot6d":
            return self.num_joints * 6
        raise ValueError(f"Unknown representation mode: {self.repr_mode}")

    def _encode_representation(self, angles: torch.Tensor) -> torch.Tensor:
        if self.repr_mode == "angle":
            return angles
        axes = self.joint_axes.to(dtype=angles.dtype)
        axis_angle = angles_to_axis_angle(angles, axes)
        if self.repr_mode == "axis_angle":
            return axis_angle.reshape(angles.shape[0], -1)
        axis_angle_flat = axis_angle.reshape(-1, 3)
        rot6d = axis_angle_to_rot6d(axis_angle_flat)
        return rot6d.reshape(angles.shape[0], -1)

    def _decode_to_angles(self, repr_tensor: torch.Tensor) -> torch.Tensor:
        if self.repr_mode == "angle":
            return repr_tensor
        axes = self.joint_axes.to(dtype=repr_tensor.dtype)
        if self.repr_mode == "axis_angle":
            axis_angle = repr_tensor.reshape(-1, self.num_joints, 3)
        elif self.repr_mode == "rot6d":
            rot6d = repr_tensor.reshape(-1, self.num_joints, 6)
            axis_angle = rot6d_to_axis_angle(rot6d.reshape(-1, 6)).reshape(
                -1, self.num_joints, 3
            )
        else:
            raise ValueError(f"Unknown representation mode: {self.repr_mode}")
        return axis_angle_to_angles(axis_angle, axes)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        joint_angles = batch['joint_angles']
        target = joint_angles
        B = batch['emg'].shape[0]
        mask = batch['label_valid_mask']
        model_input = joint_angles
        input_mode = getattr(self.model, "input_mode", "bct")
        if input_mode == "joint_5x4":
            model_input = joint_angles.reshape(B, 5, 4, joint_angles.shape[-1])
        elif input_mode != "bct":
            raise ValueError(f"Unsupported VQ-VAE input_mode: {input_mode}")

        recon, indices, vq_loss, codebook_loss, commit_loss = self.forward(model_input)
        recon_loss = F.mse_loss(recon, target)
        loss = recon_loss + vq_loss

        perplexity = VectorQuantizer.compute_perplexity(
            indices, num_codes=self.model.quantizer.num_codes
        )
        angle_mae = F.l1_loss(recon, target)

        metrics = {}
        for metric in self.metrics:
            # import ipdb;ipdb.set_trace()
            metrics.update(metric(recon, target, mask, stage))
        self.log_dict(metrics, sync_dist=True)
        
        angle_mae_deg = angle_mae * (180.0 / torch.pi)

        self.log(f"{stage}/loss", loss, sync_dist=True)
        self.log(f"{stage}/recon_loss", recon_loss, sync_dist=True)
        self.log(f"{stage}/vq_loss", vq_loss, sync_dist=True)
        self.log(f"{stage}/codebook_loss", codebook_loss, sync_dist=True)
        self.log(f"{stage}/commit_loss", commit_loss, sync_dist=True)
        self.log(f"{stage}/perplexity", perplexity, sync_dist=True)
        self.log(f"{stage}/angle_mae_rad", angle_mae, sync_dist=True)
        self.log(f"{stage}/angle_mae_deg", angle_mae_deg, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )
