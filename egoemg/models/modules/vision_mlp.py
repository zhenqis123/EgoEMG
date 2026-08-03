"""Pure vision baseline: cached ViT feature → MLP → joint angles.

No EMG pathway, no transformer decoder — the simplest possible vision baseline.
"""

from __future__ import annotations

import torch
from torch import nn


class VisionMLP(nn.Module):
    """Single-frame vision baseline: ViT feature → MLP → 22 joint angles."""

    def __init__(
        self,
        vision_embed_dim: int = 1280,
        hidden_dims: list[int] | None = None,
        out_channels: int = 22,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        layers = []
        in_dim = vision_embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(int(in_dim), int(h)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
            in_dim = h
        layers.append(nn.Linear(int(in_dim), int(out_channels)))
        self.mlp = nn.Sequential(*layers)
        self.out_channels = int(out_channels)

    @property
    def left_context(self):
        return 0

    @property
    def right_context(self):
        return 0

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vf = batch["vision_features"]  # (B, 1280)
        pred = self.mlp(vf)  # (B, 22)
        pred = pred.unsqueeze(-1)  # (B, 22, 1)

        joint_angles = batch["joint_angles"]  # (B, 22, T)
        center = joint_angles.shape[-1] // 2
        targets = joint_angles[:, :, center:center + 1]  # (B, 22, 1)

        mask = torch.ones(pred.shape[0], 1, device=pred.device, dtype=torch.float32)
        return pred, targets, mask

    @staticmethod
    def align_mask(mask: torch.Tensor, n_time: int) -> torch.Tensor:
        # Single-frame model: mask is already (B, 1), no-op if n_time==1
        return mask
