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
        input_mode = getattr(self.vqvae, "input_mode", "")
        if input_mode == "joint_5x4":
            if joint_angles.shape[1] != 20:
                raise ValueError(
                    f"Expected joint_angles with 20 channels, got {joint_angles.shape}"
                )
            b, _, t = joint_angles.shape
            x_3d = joint_angles.reshape(b, 5, 4, t)
            z_e = self.vqvae.encoder(x_3d)
            _, indices, _, _, _ = self.vqvae.quantizer(z_e)
            return indices
        b, c, _ = joint_angles.shape
        x_3d = joint_angles.transpose(1, 2).contiguous()
        x_3d = x_3d.view(-1, c)
        z_e = self.vqvae.encoder(x_3d)
        _, indices, _, _, _ = self.vqvae.quantizer(z_e)
        return indices

    def _get_num_groups(self) -> int:
        num_groups = getattr(self.model, "num_groups", None)
        if num_groups is None:
            raise ValueError("EmgToVQVAELatent must expose num_groups.")
        return int(num_groups)

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

    def _decode_stepwise_embeddings(
        self, quantized: torch.Tensor, target_t: int
    ) -> torch.Tensor:
        if quantized.ndim == 3:
            b, t, d = quantized.shape
            flat = quantized.reshape(b * t, d)
        else:
            raise ValueError(f"Expected quantized (B, T, D), got {quantized.shape}")
        recon_flat = self.vqvae.decoder(flat)
        recon = recon_flat.view(b, t, -1).transpose(1, 2).contiguous()
        if recon.shape[-1] != target_t:
            recon = F.interpolate(recon, size=target_t, mode="linear", align_corners=True)
        return recon

    def _decode_from_logits(self, logits: torch.Tensor, target_t: int) -> torch.Tensor:
        probs = F.gumbel_softmax(
            logits, tau=self.gumbel_tau, hard=self.gumbel_hard, dim=-1
        )
        if hasattr(self.vqvae, "decode_embeddings"):
            quant_sum = self._decode_soft_embeddings(probs)
            return self.vqvae.decode_embeddings(quant_sum, target_t=target_t)
        codebook = self.vqvae.quantizer.get_codebook()
        if codebook.ndim == 2:
            quantized = torch.einsum("blk,kd->bld", probs, codebook)
        elif codebook.ndim == 3:
            num_groups = codebook.shape[0]
            if probs.shape[1] % num_groups != 0:
                raise ValueError(
                    f"Levels {probs.shape[1]} not divisible by groups {num_groups}"
                )
            steps = probs.shape[1] // num_groups
            probs = probs.view(probs.shape[0], num_groups, steps, probs.shape[-1])
            quant_levels = []
            for g in range(num_groups):
                weights = probs[:, g]  # (B, T, K)
                quant_levels.append(torch.einsum("btk,kd->btd", weights, codebook[g]))
            quantized = torch.cat(quant_levels, dim=-1)
        else:
            raise ValueError(f"Unsupported codebook shape: {codebook.shape}")
        if getattr(self.vqvae, "input_mode", "") == "bct":
            return self._decode_stepwise_embeddings(quantized, target_t)
        b, l, d = quantized.shape
        num_groups = self._get_num_groups()
        if l % num_groups != 0:
            raise ValueError(
                f"Quantized length {l} not divisible by num_groups {num_groups}"
            )
        quantized = (
            quantized.view(b, num_groups, l // num_groups, d)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
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
        codebook = self.vqvae.quantizer.get_codebook()
        if codebook.ndim == 3:
            num_groups = codebook.shape[0]
            if indices.shape[1] == num_groups:
                indices_btG = indices.permute(0, 2, 1).contiguous()
            else:
                indices_btG = indices
            b, t, _ = indices_btG.shape
            quantized_flat = self.vqvae.quantizer.decode_indices(
                indices_btG.reshape(-1, num_groups)
            )
            quantized = quantized_flat.view(b, t, -1)
            if getattr(self.vqvae, "input_mode", "") == "bct":
                return self._decode_stepwise_embeddings(quantized, target_t)
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
    def _map_out_to_in_last(self, T_in: int, T_out: int, device):
        # 输出步 t_out 对齐到输入区间的最后一帧 r
        scale = T_in / T_out
        t = torch.arange(T_out, device=device)
        r = torch.floor((t + 1) * scale).long() - 1
        return r.clamp(0, T_in - 1)  # (T_out,)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        emg = batch["emg"]
        # import ipdb;ipdb.set_trace()
        logits = self.model(emg)  # (B, output_steps*num_groups, num_codes)
        joint_angles = batch["joint_angles"]  # (B, 20, input_steps)
        
        mask = batch.get("label_valid_mask") # (B, input_steps)
        B = emg.shape[0]
        G = self._get_num_groups()
        C = logits.shape[-1]

        T_in = joint_angles.shape[-1]
        T_out = logits.shape[1] // G
        assert logits.shape[1] == T_out * G, "logits second dim must be T_out*num_groups"

        # 1) 计算对齐索引：每个输出步对齐到输入的 last frame
        idx_in = self._map_out_to_in_last(T_in, T_out, device=joint_angles.device)  # (T_out,)
        # import ipdb;ipdb.set_trace()
        input_mode = getattr(self.vqvae, "input_mode", "")
        with torch.no_grad():
            if input_mode == "joint_5x4":
                indices_bgT = self._compute_target_indices(joint_angles)
                if indices_bgT.shape[1] != G:
                    raise ValueError(
                        f"Expected indices groups {G}, got {indices_bgT.shape}"
                    )
                if indices_bgT.shape[-1] != T_out:
                    idx_vq = self._map_out_to_in_last(
                        indices_bgT.shape[-1], T_out, device=joint_angles.device
                    )
                    indices_bgT = indices_bgT[:, :, idx_vq]
                indices_btG = indices_bgT.permute(0, 2, 1).contiguous()
            else:
                # 2) 只抽取这些输入帧的真值（只对这些时间步算 indices）
                joint_ds = joint_angles[:, :, idx_in]  # (B, 20, T_out)
                # 3) 目标 indices 只按 T_out 计算
                indices = self._compute_target_indices(joint_ds)  # (B*T_out, G)
                indices_btG = indices.view(B, T_out, G)
                indices_bgT = indices_btG.permute(0, 2, 1).contiguous()

        # 4) logits reshape 到 (B, T_out, G, C) 再 flatten
        logits_btgc = logits.view(B, T_out, G, C)              # (B, T_out, G, C)
        logits_flat = logits_btgc.reshape(-1, C)               # (B*T_out*G, C)

        # 5) targets flatten 到 (B*T_out*G,)
        targets_flat = indices_btG.reshape(-1)   # (B*T_out*G,)

        # 6) mask 同步抽取到 T_out，并扩展到 group 维
        ce_loss = torch.tensor(0.0, device=logits.device)
        if mask is not None:
            mask_ds = mask[:, idx_in].to(torch.bool)                # (B, T_out)
            valid = mask_ds[:, :, None].expand(B, T_out, G).reshape(-1)  # (B*T_out*G,)

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
        # import ipdb;ipdb.set_trace()
        recon = self._decode_from_logits(logits, joint_angles.shape[-1])
        recon_q = self._decode_indices(indices_bgT, joint_angles.shape[-1])
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

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return instantiate_optimizer_and_scheduler(
            self.model.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )
