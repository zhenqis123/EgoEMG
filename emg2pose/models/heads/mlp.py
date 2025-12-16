from typing import Iterable, Literal, Sequence

import torch
from torch import nn


def _get_activation(name: Literal["relu", "gelu"]) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class MLPHead(nn.Module):
    """Simple MLP head mapping sequence features (B, C, T) to predictions (B, out, T)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_sizes: Sequence[int] | Iterable[int] = (512,),
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        act = _get_activation(activation)

        layers: list[nn.Module] = []
        prev = in_channels
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = size

        layers.append(nn.Linear(prev, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, _, t = x.shape
        x = x.transpose(1, 2).contiguous().view(b * t, -1)  # (B*T, C)
        out = self.net(x)
        out = out.view(b, t, -1).transpose(1, 2)  # (B, out_channels, T)
        return out
