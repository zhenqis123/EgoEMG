# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class MLP(nn.Module):
    """Basic MLP decoder with optional output scaling."""

    def __init__(
        self,
        in_channels: int,
        layer_sizes: list[int],
        out_channels: int,
        layer_norm: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()

        sizes = [in_channels] + layer_sizes
        layers: list[nn.Module] = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if layer_norm:
                layers.append(nn.LayerNorm(out_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(sizes[-1], out_channels))

        self.mlp = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) * self.scale
