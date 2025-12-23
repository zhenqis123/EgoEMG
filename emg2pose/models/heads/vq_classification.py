from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from emg2pose.models.quantizers.vq import (
    GroupedVectorQuantizer,
    ResidualVectorQuantizer,
    VectorQuantizer,
)


class VQDiscreteHead(nn.Module):
    """
    Classification head predicting code indices for residual VQ levels.
    Decoding is delegated to a loaded JointAngleVQVAE module.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_codes: int,
        num_levels: int = 1,
        num_joints: int = 20,
        angle_representation: Literal["angle", "axis_angle", "rot6d"] = "angle",
        checkpoint: str | None = None,
        freeze_codebook: bool = True,
        freeze_decoder: bool = True,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        if checkpoint is None:
            raise ValueError("VQDiscreteHead requires a VQ-VAE checkpoint.")
        self.vqvae_module = self._load_vqvae_module(
            checkpoint,
            expected_num_levels=num_levels,
            expected_num_codes=num_codes,
            expected_embed_dim=embed_dim,
            expected_num_joints=num_joints,
            expected_repr=angle_representation,
            freeze_codebook=freeze_codebook,
            freeze_decoder=freeze_decoder,
            freeze_encoder=freeze_encoder,
        )
        self.vqvae_module.eval()
        quantizer = self.vqvae_module.model.quantizer
        model_levels = getattr(self.vqvae_module.model, "num_index_levels", None)
        self.num_levels = int(model_levels) if model_levels is not None else int(
            getattr(quantizer, "num_groups", 1)
        )
        self.num_codes = int(getattr(quantizer, "num_codes", -1))
        self.num_joints = int(getattr(self.vqvae_module, "num_joints", num_joints))
        self.angle_representation = str(getattr(self.vqvae_module, "repr_mode", "angle"))

        self.proj = nn.Linear(in_channels, embed_dim)
        self.logits_heads = nn.ModuleList(
            [nn.Linear(embed_dim, self.num_codes) for _ in range(self.num_levels)]
        )

    def _repr_to_angles(self, repr_tensor: torch.Tensor) -> torch.Tensor:
        return self.vqvae_module._decode_to_angles(repr_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            logits: (B, num_codes, T, num_levels)
        """
        b, _, t = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(b * t, -1)  # (B*T, C)
        h = self.proj(x_flat)
        logits_levels = []
        for head in self.logits_heads:
            logit = head(h)  # (B*T, num_codes)
            logits_levels.append(logit.view(b, t, self.num_codes))
        logits = torch.stack(logits_levels, dim=-1)  # (B, t, num_codes, L)
        logits = logits.permute(0, 2, 1, 3)  # (B, num_codes, T, L)
        return logits  # (B, num_codes, T, L)

    def train(self, mode: bool = True) -> "VQDiscreteHead":
        super().train(mode)
        self.vqvae_module.eval()
        return self

    def decode_from_logits(
        self, logits: torch.Tensor, target_t: int | None = None
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_codes, T, L)
        Returns:
            angles: (B, num_joints, T)
        """
        b, _, t, L = logits.shape
        pred_idx = logits.argmax(dim=1)  # (B, T, L)
        return self.decode_from_indices(pred_idx, target_t=target_t)

    def decode_from_logits_gumbel(
        self,
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        target_t: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_codes, T, L)
        Returns:
            angles: (B, num_joints, T)
        """
        b, _, t, _ = logits.shape
        if not hasattr(self.vqvae_module.model, "decode_embeddings"):
            raise RuntimeError(
                "VQDiscreteHead requires vqvae.model.decode_embeddings for gumbel decode."
            )
        probs = F.gumbel_softmax(
            logits.permute(0, 3, 2, 1), tau=tau, hard=hard, dim=-1
        )  # (B, L, T, K)
        quant_sum = self._decode_soft_embeddings(probs)
        return self.vqvae_module.model.decode_embeddings(
            quant_sum, target_t=target_t or t
        )

    def _decode_soft_embeddings(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: (B, L, T, K)
        Returns:
            quantized: (B, T, D)
        """
        quantizer = self.vqvae_module.model.quantizer
        codebook = quantizer.get_codebook()
        if isinstance(quantizer, VectorQuantizer):
            if probs.shape[1] == 1:
                weights = probs[:, 0]  # (B, T, K)
                return torch.einsum("btk,kd->btd", weights, codebook)
            if getattr(self.vqvae_module.model, "per_joint_quantization", False):
                quant_levels = []
                for l in range(probs.shape[1]):
                    weights = probs[:, l]  # (B, T, K)
                    quant_levels.append(torch.einsum("btk,kd->btd", weights, codebook))
                return torch.cat(quant_levels, dim=-1)
            raise ValueError(
                f"Expected one level for VectorQuantizer, got {probs.shape[1]}"
            )
        if isinstance(quantizer, GroupedVectorQuantizer):
            if codebook.shape[0] != probs.shape[1]:
                raise ValueError(
                    f"Levels {probs.shape[1]} != codebook groups {codebook.shape[0]}"
                )
            quant_levels = []
            for g in range(probs.shape[1]):
                weights = probs[:, g]  # (B, T, K)
                quant_levels.append(torch.einsum("btk,kd->btd", weights, codebook[g]))
            return torch.cat(quant_levels, dim=-1)
        if isinstance(quantizer, ResidualVectorQuantizer):
            if codebook.shape[0] != probs.shape[1]:
                raise ValueError(
                    f"Levels {probs.shape[1]} != codebook levels {codebook.shape[0]}"
                )
            quant_sum = torch.zeros(
                probs.shape[0],
                probs.shape[2],
                codebook.shape[-1],
                device=probs.device,
                dtype=probs.dtype,
            )
            for l in range(probs.shape[1]):
                weights = probs[:, l]  # (B, T, K)
                quant_sum = quant_sum + torch.einsum("btk,kd->btd", weights, codebook[l])
            return quant_sum
        raise TypeError(f"Unsupported quantizer type: {type(quantizer)}")

    def decode_from_indices(
        self, indices: torch.Tensor, target_t: int | None = None
    ) -> torch.Tensor:
        """
        Args:
            indices: (B, T, L)
        Returns:
            angles: (B, num_joints, T)
        """
        b, t, L = indices.shape
        if not hasattr(self.vqvae_module.model, "decode_indices"):
            raise RuntimeError(
                "VQDiscreteHead requires vqvae.model.decode_indices for decode."
            )
        indices_btj = indices.permute(0, 2, 1).contiguous()
        return self.vqvae_module.model.decode_indices(
            indices_btj, target_t=target_t or t
        )

    def _load_vqvae_module(
        self,
        checkpoint: str,
        expected_num_levels: int,
        expected_num_codes: int,
        expected_embed_dim: int,
        expected_num_joints: int,
        expected_repr: str,
        freeze_codebook: bool = True,
        freeze_decoder: bool = True,
        freeze_encoder: bool = True,
    ) -> nn.Module:
        from emg2pose.lightning_vqvae import JointAngleVQVAEModule

        ckpt = JointAngleVQVAEModule.load_from_checkpoint(
            checkpoint, map_location="cpu", strict=True
        )

        # -------------------------------------------------
        # Force-freeze ALL parameters (encoder + codebook + decoder)
        # -------------------------------------------------
        for p in ckpt.parameters():
            p.requires_grad = False

        ckpt.eval()  # optional but strongly recommended

        return ckpt
