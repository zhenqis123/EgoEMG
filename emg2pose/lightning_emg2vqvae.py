from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import functional as F

import pytorch_lightning as pl

from emg2pose.metrics import get_default_metrics
from emg2pose.models.modules.vqvae_vqmyo import JointAngleVQVAE
from emg2pose.utils import instantiate_optimizer_and_scheduler

log = logging.getLogger(__name__)


class EmgToVQVAELightningModule(pl.LightningModule):
    def __init__(
        self,
        model_conf: DictConfig,
        vqvae_conf: DictConfig,
        vqvae_checkpoint: str,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig | None,
        gumbel_conf: DictConfig | None = None,
        loss_weights: DictConfig | None = None,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(model_conf, _convert_="all")
        self.vqvae = self._load_vqvae(vqvae_conf, vqvae_checkpoint)
        self.vqvae.eval()
        self._freeze_vqvae()

        self.gumbel_conf = gumbel_conf or {}
        self.gumbel_tau = float(self.gumbel_conf.get("temperature", 1.0))
        self.gumbel_hard = bool(self.gumbel_conf.get("hard", False))

        self.loss_weights = loss_weights or {"latent": 1.0, "recon": 0.5}
        self.label_smoothing = float(label_smoothing)
        self.metrics = get_default_metrics()

    def train(self, mode: bool = True) -> "EmgToVQVAELightningModule":
        super().train(mode)
        self.vqvae.eval()
        return self

    def _freeze_vqvae(self) -> None:
        for p in self.vqvae.parameters():
            p.requires_grad = False

    def _load_vqvae(self, vqvae_conf: DictConfig, ckpt_path: str) -> JointAngleVQVAE:
        model = instantiate(vqvae_conf, _convert_="all")
        ckpt = torch.load(Path(ckpt_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        filtered = {
            k.removeprefix("model."): v
            for k, v in state.items()
            if k.startswith("model.")
        }
        if not filtered:
            filtered = state
        missing, unexpected = model.load_state_dict(filtered, strict=True)
        if missing:
            log.warning("Missing VQ-VAE keys: %s", missing)
        if unexpected:
            log.warning("Unexpected VQ-VAE keys: %s", unexpected)
        return model

    def _compute_target_indices(self, joint_angles: torch.Tensor) -> torch.Tensor:
        b, c, t = joint_angles.shape
        if c != 20:
            raise ValueError(f"Expected 20 joint channels, got {c}")
        if hasattr(self.vqvae, "quantize_angles"):
            return self.vqvae.quantize_angles(joint_angles)
        if t != 2000:
            raise ValueError(f"Expected T=2000 for VQ-VAE, got {t}")
        x_3d = joint_angles.view(b, 5, 4, t)
        z_e = self.vqvae.encoder(x_3d)
        _, indices, _, _, _ = self.vqvae.quantizer(z_e)
        return indices  # (B, 5, 200)

    def _downsample_mask(self, mask: torch.Tensor, steps: int = 200) -> torch.Tensor:
        t = mask.shape[-1]
        positions = torch.round(
            torch.linspace(0, t - 1, steps=steps, device=mask.device)
        ).long()
        positions = torch.clamp(positions, 0, t - 1)
        return mask[:, positions]

    def _decode_soft_embeddings(self, probs: torch.Tensor) -> torch.Tensor:
        quantizer = self.vqvae.quantizer
        codebook = quantizer.get_codebook()
        if probs.ndim == 3:
            # probs: (B, L, K)
            if probs.shape[1] == 1:
                weights = probs[:, 0:1]
                return torch.einsum("blk,kd->bld", weights, codebook)
            if getattr(self.vqvae, "per_joint_quantization", False):
                h = int(getattr(self.vqvae, "num_index_levels", 1))
                if probs.shape[1] % h != 0:
                    raise ValueError(
                        f"Expected L divisible by H, got L={probs.shape[1]}, H={h}"
                    )
                w = probs.shape[1] // h
                probs_hw = probs.view(probs.shape[0], h, w, probs.shape[-1])
                quant_levels = []
                for level in range(h):
                    weights = probs_hw[:, level]  # (B, W, K)
                    quant_levels.append(torch.einsum("bwk,kd->bwd", weights, codebook))
                return torch.cat(quant_levels, dim=-1)  # (B, W, H*D)
            return torch.einsum("blk,kd->bld", probs, codebook)
        if probs.ndim == 4:
            # probs: (B, L, T, K)
            if probs.shape[1] == 1:
                weights = probs[:, 0]
                return torch.einsum("btk,kd->btd", weights, codebook)
            if getattr(self.vqvae, "per_joint_quantization", False):
                quant_levels = []
                for level in range(probs.shape[1]):
                    weights = probs[:, level]
                    quant_levels.append(torch.einsum("btk,kd->btd", weights, codebook))
                return torch.cat(quant_levels, dim=-1)
        raise ValueError(f"Unsupported probs shape for quantizer: {probs.shape}")

    def _decode_from_logits(self, logits: torch.Tensor, target_t: int) -> torch.Tensor:
        probs = F.gumbel_softmax(
            logits, tau=self.gumbel_tau, hard=self.gumbel_hard, dim=-1
        )
        if hasattr(self.vqvae, "decode_embeddings"):
            quant_sum = self._decode_soft_embeddings(probs)
            return self.vqvae.decode_embeddings(quant_sum, target_t=target_t)
        codebook = self.vqvae.quantizer.vq.codebook.weight
        quantized = torch.einsum("blk,kd->bld", probs, codebook)
        b, l, d = quantized.shape
        quantized = quantized.view(b, 5, l // 5, d).permute(0, 3, 1, 2).contiguous()
        return self._decode_quantized(quantized, target_t)

    def _decode_quantized(self, quantized: torch.Tensor, target_t: int) -> torch.Tensor:
        x_hat_3d = self.vqvae.decoder(quantized, target_T=target_t)
        b, h, c, t = x_hat_3d.shape
        recon = x_hat_3d.view(b, h * c, t)
        recon = self.vqvae.mixing(recon)
        return recon

    def _decode_indices(self, indices: torch.Tensor, target_t: int) -> torch.Tensor:
        if indices.ndim != 3:
            raise ValueError(f"Expected indices (B, H, W), got {tuple(indices.shape)}")
        if hasattr(self.vqvae, "decode_indices"):
            return self.vqvae.decode_indices(indices, target_t=target_t)
        codebook = self.vqvae.quantizer.vq.codebook.weight
        b, h, w = indices.shape
        flat = indices.reshape(-1)
        quantized = codebook.index_select(0, flat).view(b, h, w, -1)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self._decode_quantized(quantized, target_t)

    def _masked_mse(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        diff = (pred - target) ** 2
        if mask is None:
            return diff.mean()
        denom = mask.sum().clamp(min=1) * diff.shape[1]
        return (diff * mask[:, None, :]).sum() / denom

    def _masked_mae(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)
        if mask is None:
            return diff.mean()
        denom = mask.sum().clamp(min=1) * diff.shape[1]
        return (diff * mask[:, None, :]).sum() / denom

    def build_valid_mask(
        self, base_mask: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        mask = base_mask.bool()
        finite = torch.isfinite(targets).all(dim=1)
        mask = mask & finite
        if mask.sum() == 0:
            log.warning(
                "All samples masked out after combining IK/interp/finite checks."
            )
        return mask

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        emg = batch["emg"]
        # import ipdb;ipdb.set_trace()
        logits = self.model(emg)  # (B, 1000, K)
        if logits.ndim != 3:
            raise ValueError(f"Expected logits (B, 1000, K), got {tuple(logits.shape)}")

        joint_angles = batch["joint_angles"]  # (B, 20, 2000)
        mask = batch.get("label_valid_mask")

        with torch.no_grad():
            indices = self._compute_target_indices(joint_angles)  # (B, 5, 200)

        targets_flat = indices.reshape(indices.shape[0], -1)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets_flat.reshape(-1)

        ce_loss = torch.tensor(0.0, device=logits.device)
        if mask is not None:
            mask_sub = self._downsample_mask(mask, steps=indices.shape[-1])
            mask_flat = mask_sub[:, None, :].expand_as(indices).reshape(-1)
            valid = mask_flat.bool()
            if valid.any():
                ce_loss = F.cross_entropy(
                    logits_flat[valid],
                    targets_flat[valid],
                    label_smoothing=self.label_smoothing,
                )
        else:
            ce_loss = F.cross_entropy(
                logits_flat, targets_flat, label_smoothing=self.label_smoothing
            )

        recon = self._decode_from_logits(logits, joint_angles.shape[-1])
        recon_q = self._decode_indices(indices, joint_angles.shape[-1])
        valid_mask = (
            self.build_valid_mask(mask, joint_angles) if mask is not None else None
        )
        recon_loss = self._masked_mse(recon, joint_angles, valid_mask)
        recon_mae = self._masked_mae(recon, joint_angles, valid_mask)
        recon_q_mse = self._masked_mse(recon_q, joint_angles, valid_mask)
        recon_q_mae = torch.sqrt(torch.clamp(recon_q_mse, min=1e-12))

        loss = self.loss_weights.get("latent", 1.0) * ce_loss + self.loss_weights.get(
            "recon", 0.5
        ) * recon_loss

        metrics = {}
        if valid_mask is not None:
            for metric in self.metrics:
                metrics.update(metric(recon, joint_angles, valid_mask, stage))
            self.log_dict(metrics, sync_dist=True)

        self.log(f"{stage}/loss", loss, sync_dist=True)
        self.log(f"{stage}/latent_ce", ce_loss, sync_dist=True)
        self.log(f"{stage}/recon_mse", recon_loss, sync_dist=True)
        self.log(f"{stage}/recon_mae", recon_mae, sync_dist=True)
        self.log(
            f"{stage}/recon_mae_deg",
            recon_mae * (180.0 / torch.pi),
            sync_dist=True,
        )
        self.log(f"{stage}/vqvae_recon_mse", recon_q_mse, sync_dist=True)
        self.log(f"{stage}/vqvae_recon_mae", recon_q_mae, sync_dist=True)
        self.log(
            f"{stage}/vqvae_recon_mae_deg",
            recon_q_mae * (180.0 / torch.pi),
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return instantiate_optimizer_and_scheduler(
            self.model.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )
