"""Shared temporal attention pooling used by EMGFormer + MidFusionPoseFormer.

Both modules need to collapse a (B, C, T) feature sequence into (B, C) via
learned per-timestep attention weights (to pick which time steps inform the
center-frame prediction). Previously the same ~10-line nn.Sequential + 4-line
forward block was duplicated in emgformer.py and mid_fusion.py.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPool(nn.Module):
    """(B, C, T) → (B, C) via learned temporal attention.

    Computes a per-timestep score (C → hidden → 1), softmaxes over time, and
    returns the attention-weighted sum across the time dimension.
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        hidden = max(int(feat_dim) // 4, 16)
        self.score = nn.Sequential(
            nn.Linear(int(feat_dim), hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → returns (B, C)."""
        attn_scores = self.score(x.transpose(1, 2))  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        return (x * attn_weights.squeeze(-1).unsqueeze(1)).sum(dim=-1)  # (B, C)
