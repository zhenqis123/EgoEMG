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
from emg2pose.datasets.multisession_emg2pose_dataset import (
    MultiSessionWindowedEmgDataset,
)
from emg2pose.metrics import get_default_metrics
from emg2pose.models.modules import BaseModule
from hydra.utils import instantiate

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
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
        max_open_files: int = 32,
        norm_mode: str | None = None,
        norm_stats_path: str | None = None,
        norm_eps: float = 1e-6,
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
        self.max_open_files = max_open_files
        self.norm_mode = norm_mode
        self.norm_stats_path = norm_stats_path
        self.norm_eps = norm_eps

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.skip_ik_failures = skip_ik_failures

    def setup(self, stage: str | None = None) -> None:
        # train
        self.train_dataset = MultiSessionWindowedEmgDataset(
            hdf5_paths=list(self.train_sessions),
            transform=self.train_transforms,
            window_length=self.window_length,
            stride=self.stride,
            padding=self.padding,
            jitter=True,
            skip_ik_failures=self.skip_ik_failures,
            max_open_files=self.max_open_files,
            norm_mode=self.norm_mode,
            norm_stats_path=self.norm_stats_path,
            norm_eps=self.norm_eps,
        )

        # val
        self.val_dataset = MultiSessionWindowedEmgDataset(
            hdf5_paths=list(self.val_sessions),
            transform=self.val_transforms,
            window_length=self.val_test_window_length,
            stride=self.val_test_stride,
            padding=self.padding,
            jitter=False,
            skip_ik_failures=self.skip_ik_failures,
            max_open_files=self.max_open_files,
            norm_mode=self.norm_mode,
            norm_stats_path=self.norm_stats_path,
            norm_eps=self.norm_eps,
        )

        # test
        self.test_dataset = MultiSessionWindowedEmgDataset(
            hdf5_paths=list(self.test_sessions),
            transform=self.test_transforms,
            window_length=self.val_test_window_length,
            stride=self.val_test_stride,
            padding=(0, 0),
            jitter=False,
            skip_ik_failures=self.skip_ik_failures,
            max_open_files=self.max_open_files,
            norm_mode=self.norm_mode,
            norm_stats_path=self.norm_stats_path,
            norm_eps=self.norm_eps,
        )


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
        task_type: str = "regression",  # regression | discrete
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        gumbel_recon: DictConfig | None = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.model: BaseModule = instantiate(module_conf, _convert_="all")
        self.provide_initial_pos = bool(getattr(self.model, "provide_initial_pos", False))
        self.loss_weights = loss_weights or {"mae": 1}
        self._warned_emg_nan = False
        self.task_type = task_type
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)
        # import ipdb;ipdb.set_trace()
        self.gumbel_recon = gumbel_recon or {}
        # self.use_gumbel_recon = bool(self.gumbel_recon.get("enabled", False))
        self.use_gumbel_recon = True
        self.gumbel_tau = float(self.gumbel_recon.get("temperature", 1.0))
        self.gumbel_hard = bool(self.gumbel_recon.get("hard", False))
        self.gumbel_weight = float(self.gumbel_recon.get("weight", 1.0))
        self._warned_gumbel_unfrozen = False

        # Metrics sets
        self.regression_metrics = get_default_metrics()
        self.discrete_metrics: list = []  # placeholder for custom discrete metrics if needed

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # import ipdb;ipdb.set_trace()
        out = self.model.forward(batch, self.provide_initial_pos)
        if self.task_type == "discrete":
            return self._prepare_discrete(out, batch)
        preds = out
        joint_angles = batch["joint_angles"]
        mask = batch["label_valid_mask"]
        start = self.model.left_context
        stop = None if self.model.right_context == 0 else -self.model.right_context
        targets = joint_angles[..., slice(start, stop)]
        mask = mask[..., slice(start, stop)]
        if preds.ndim == 2:
            preds = preds[..., None]
        n_time = targets.shape[-1]
        preds = self.model.align_predictions(preds, n_time)
        mask = self.model.align_mask(mask, n_time)

        return preds, targets, mask

    def _prepare_discrete(
        self, preds: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_angles = batch["joint_angles"]
        start = self.model.left_context
        stop = None if self.model.right_context == 0 else -self.model.right_context
        targets_full = joint_angles[..., slice(start, stop)]
        code_indices = self._quantize_angles(targets_full)  # (B, L, T_code)
        logits = preds.permute(0, 3, 2, 1).contiguous()  # (B, L, T_pred, num_codes)
        return logits, code_indices

    def _quantize_angles(self, joint_angles: torch.Tensor) -> torch.Tensor:
        vqvae = self.model.head.vqvae_module
        b, j, t = joint_angles.shape
        with torch.no_grad():
            if hasattr(vqvae.model, "quantize_angles"):
                return vqvae.model.quantize_angles(joint_angles)
            flat = joint_angles.transpose(1, 2).reshape(-1, j)
            repr_in = vqvae._encode_representation(flat)
            z_e = vqvae.model.encoder(repr_in)
            _, indices, _, _, _ = vqvae.model.quantizer(z_e)
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
        if indices.shape[0] == flat.shape[0]:
            indices = indices.transpose(0, 1)
        return indices.view(indices.shape[0], b, t).permute(1, 0, 2)

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str = "train"
    ) -> torch.Tensor:

        # Generate predictions
        if getattr(self.hparams, "datamodule", None) and self.hparams.datamodule.get("norm_mode") == "batch":
            emg = batch["emg"]
            mean = emg.mean()
            std = emg.std()
            batch["emg"] = (emg - mean) / (std + 1e-6)
        preds, targets= self.forward(batch) 
        joint_angles = batch['joint_angles']
        start = self.model.left_context
        stop = None if self.model.right_context == 0 else -self.model.right_context
        joint_angle_targets = joint_angles[..., slice(start, stop)]
        n_time = joint_angle_targets.shape[-1]
        if self.task_type == "discrete":
            code_sub = targets
            cls_loss = self._discrete_loss(preds, code_sub, None, stage)
            cls_weight = self.loss_weights.get("cls", 1.0)
            loss = cls_loss * cls_weight
            # import ipdb;ipdb.set_trace()
            if self.use_gumbel_recon:
                gumbel_loss = self._gumbel_recon_loss(preds, code_sub, None, stage)
                loss = loss + gumbel_loss * self.gumbel_weight
            # import ipdb;ipdb.set_trace()
            decoded_angles = self.model.head.decode_from_logits(
                preds.permute(0, 3, 2, 1), target_t=n_time
            )  # (B, J, T_pred)
            if decoded_angles.ndim == 2:
                decoded_angles = decoded_angles[..., None]

            mask_full = batch["label_valid_mask"][..., slice(start, stop)]
            mask_aligned = self.model.align_mask(mask_full, n_time)
            # align targets_full to match decoded_aligned length
            valid_mask = mask_aligned.bool()
            metrics = {}
            
            for metric in self.regression_metrics:
                metrics.update(metric(decoded_angles, joint_angle_targets, valid_mask, stage))
            self.log_dict(metrics, sync_dist=True)
            vqvae = self.model.head.vqvae_module
            if hasattr(vqvae.model, "quantize_angles"):
                recon_angles, _, _, _, _ = vqvae(joint_angle_targets)
            else:
                recon_angles, _, _, _, _ = vqvae(joint_angle_targets.reshape(-1, 20))
                recon_angles = recon_angles.reshape(joint_angle_targets.shape)
            recon_diff = torch.abs(recon_angles - joint_angle_targets)
            denom = valid_mask.sum() * recon_diff.shape[1]
            recon_mae = (recon_diff * valid_mask[:, None, :]).sum() / denom
            recon_mae_deg = recon_mae * (180.0 / torch.pi)
            self.log(f"{stage}_recon_mae", recon_mae, sync_dist=True)
            self.log(f"{stage}_recon_mae_deg", recon_mae_deg, sync_dist=True)
            self.log(f"{stage}_loss", loss, sync_dist=True)
            return loss

        # regression path
        valid_mask = mask.bool()
        metrics = {}
        for metric in self.regression_metrics:
            metrics.update(metric(preds, targets, valid_mask, stage))
        self.log_dict(metrics, sync_dist=True)

        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            loss += metrics.get(f"{stage}_{loss_name}", 0.0) * weight
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
        params = list(self.parameters())
        vqvae = getattr(getattr(self.model, "head", None), "vqvae_module", None)
        if vqvae is not None:
            excluded = {id(p) for p in vqvae.parameters()}
            params = [p for p in params if id(p) not in excluded]
        return utils.instantiate_optimizer_and_scheduler(
            params,
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )

    def _discrete_loss(
        self,
        logits: torch.Tensor,
        code_indices: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """
        logits: (B, L, T_pred, num_codes)
        code_indices: (B, T_pred, L)
        mask: (B, T_pred)
        """
        B, L, T_pred, num_codes = logits.shape
        code_indices = code_indices.long()

        logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, num_codes)  # (B*T_pred*L, K)
        targets_flat = code_indices.reshape(-1)  # (B*T_pred*L,)
        if mask is not None:
            mask_flat = mask.unsqueeze(-1).expand(-1, -1, L).reshape(-1)  # (B*T_pred*L,)
            valid = mask_flat.bool()
            logits_flat = logits_flat[valid]
            targets_flat = targets_flat[valid]
        ce_loss = torch.nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        # Accuracy
        with torch.no_grad():
            pred_idx = logits.argmax(dim=-1)  # (B, L, T_pred)
            # import ipdb;ipdb.set_trace()
            correct = (pred_idx == code_indices).to(torch.float32)
            # if mask is not None:
            #     correct = correct * mask.unsqueeze(-1)
            acc = correct.sum() / (mask.unsqueeze(-1).sum() + 1e-6) if mask is not None else correct.mean()

        self.log(f"{stage}_cls_ce", ce_loss, sync_dist=True)
        self.log(f"{stage}_cls_acc", acc, sync_dist=True)

        return ce_loss

    def _gumbel_recon_loss(
        self,
        logits: torch.Tensor,
        code_indices: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        self._warn_if_gumbel_unfrozen()
        logits_for_head = logits.permute(0, 3, 2, 1).contiguous()  # (B, K, T, L)
        # import ipdb;ipdb.set_trace()
        pred_angles = self.model.head.decode_from_logits_gumbel(
            logits_for_head, tau=self.gumbel_tau, hard=self.gumbel_hard
        )
        target_angles = self.model.head.decode_from_indices(code_indices.permute(0, 2, 1))
        diff = torch.abs(pred_angles - target_angles)
        if mask is not None:
            valid = mask.bool()
            denom = valid.sum().clamp(min=1) * diff.shape[1]
            loss = (diff * valid[:, None, :]).sum() / denom
        else:
            loss = diff.mean()
        self.log(f"{stage}_gumbel_recon_mae", loss, sync_dist=True)
        self.log(f"{stage}_gumbel_recon_mae_deg", loss * (180.0 / torch.pi), sync_dist=True)
        return loss

    def _warn_if_gumbel_unfrozen(self) -> None:
        if self._warned_gumbel_unfrozen or not self.use_gumbel_recon:
            return
        head = getattr(self.model, "head", None)
        if head is None or not hasattr(head, "vqvae_module"):
            self._warned_gumbel_unfrozen = True
            return
        model = head.vqvae_module.model
        quantizer = model.quantizer
        codebook_trainable = any(p.requires_grad for p in quantizer.parameters())
        decoder_modules = []
        if hasattr(model, "decoder"):
            decoder_modules.append(model.decoder)
        if hasattr(model, "upsample"):
            decoder_modules.append(model.upsample)
        if decoder_modules:
            decoder_trainable = any(
                p.requires_grad for module in decoder_modules for p in module.parameters()
            )
        else:
            decoder_trainable = False
        if codebook_trainable or decoder_trainable:
            log.warning(
                "Gumbel recon enabled but VQ codebook/decoder are trainable. "
                "Consider freeze_codebook=True and freeze_decoder=True."
            )
        self._warned_gumbel_unfrozen = True

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
