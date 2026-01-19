from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialRoPEFormer(nn.Module):
    """Featurizer + CyRoPE decoder + head for (B, C, T, D) tokenization."""

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        head: nn.Module,
        num_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.decoder = decoder
        self.head = head
        self.num_channels = int(num_channels)
        self.embed_dim = int(embed_dim)
        self.left_context = featurizer.left_context
        self.right_context = featurizer.right_context

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        feats = self.featurizer(emg)  # (B, C*D, T)
        if feats.ndim != 3:
            raise ValueError(f"Expected features to be 3D, got {feats.shape}.")
        bsz, feat_dim, t = feats.shape
        expected = self.num_channels * self.embed_dim
        if feat_dim != expected:
            raise ValueError(
                f"Feature dim {feat_dim} does not match num_channels*embed_dim {expected}."
            )
        feats = feats.view(bsz, self.num_channels, self.embed_dim, t).permute(0, 1, 3, 2)
        decoded = self.decoder(feats)  # (B, C, T, D_out)
        if decoded.ndim != 4:
            raise ValueError(f"Decoder output must be 4D, got {decoded.shape}.")
        if decoded.shape[1] != self.num_channels:
            raise ValueError(
                "Decoder output channel dim must match num_channels."
            )
        pooled = decoded.mean(dim=1)  # (B, T, D_out)
        pooled = pooled.transpose(1, 2)  # (B, D_out, T)
        return self.head(pooled)

    @staticmethod
    def align_predictions(pred: torch.Tensor, n_time: int) -> torch.Tensor:
        if pred.shape[-1] == n_time:
            return pred
        return F.interpolate(pred, size=n_time, mode="linear")

    @staticmethod
    def align_mask(mask: torch.Tensor, n_time: int) -> torch.Tensor:
        if mask.shape[-1] == n_time:
            return mask
        mask = mask[:, None].to(torch.float32)
        aligned = F.interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)
