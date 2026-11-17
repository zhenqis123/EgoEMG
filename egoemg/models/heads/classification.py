from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """Multi-task classification head with independent MLP blocks per task."""

    def __init__(
        self,
        in_channels: int,
        num_classes: Sequence[int],
        hidden_sizes: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tasks = len(num_classes)
        self.heads = nn.ModuleList()

        for k in num_classes:
            layers: list[nn.Module] = []
            prev = in_channels
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, k))
            self.heads.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, C, T).
        Returns:
            List of logits, each of shape (B, num_classes_i, T).
        """
        b, _, t = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(b * t, -1)  # (B*T, C)
        outputs: list[torch.Tensor] = []
        for head in self.heads:
            logits = head(x_flat)  # (B*T, K)
            logits = logits.view(b, t, -1).transpose(1, 2)  # (B, K, T)
            outputs.append(logits)
        return outputs
