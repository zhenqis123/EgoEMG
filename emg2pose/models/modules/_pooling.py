"""Shared temporal attention pooling used by EMGFormer + MidFusionPoseFormer.

Both modules need to collapse a (B, C, T) feature sequence into (B, C) via
learned per-timestep attention weights. Previously the same nn.Sequential +
4-line forward block was duplicated in emgformer.py and mid_fusion.py.

Backward-compat: inherits nn.Sequential so the state_dict keys
(temporal_attn.0.weight, temporal_attn.2.weight) match the old
`nn.Sequential(Linear, Tanh, Linear)` definition — checkpoints saved before
the extraction load without remapping.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPool(nn.Sequential):
    """(B, C, T) → (B, C) via learned temporal attention.

    Layers (indexed 0/1/2, matching the original nn.Sequential):
      0: Linear(feat_dim → hidden)
      1: Tanh()
      2: Linear(hidden → 1)

    forward computes attention scores over time, softmaxes, and returns the
    weighted sum across the time dimension.
    """

    def __init__(self, feat_dim: int) -> None:
        hidden = max(int(feat_dim) // 4, 16)
        super().__init__(
            nn.Linear(int(feat_dim), hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → returns (B, C)."""
        attn_scores = super().forward(x.transpose(1, 2))  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        return (x * attn_weights.squeeze(-1).unsqueeze(1)).sum(dim=-1)  # (B, C)
