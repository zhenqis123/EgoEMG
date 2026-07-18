# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class SMU(nn.Module):
    """Smooth Maximum Unit activation [Biswas et al., 2021].

    SMU(x) = (1+α)·x + (1-α)·x·erf(μ·(1-α)·x) / 2
    where α ∈ [0,1] and μ > 0.
    """

    def __init__(self, alpha: float = 1.0, mu: float = 1.0, learnable: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=learnable)
        self.mu = nn.Parameter(torch.tensor(mu), requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        mu = torch.clamp(self.mu, min=0.01)
        return (1 + alpha) * x + (1 - alpha) * x * torch.erf(mu * (1 - alpha) * x) / 2


class CircularPad2d(nn.Module):
    """Circular padding along the height (channel) dimension, zero on width (time).

    Wraps ``F.pad`` with mode='circular' on the channel dim, leaving the
    Conv2d to handle time padding via the kernel's ``padding`` argument.
    """

    def __init__(self, pad_c: int):
        super().__init__()
        self.pad_c = pad_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); H = EMG channels, W = time
        return F.pad(x, (0, 0, self.pad_c, self.pad_c), mode="circular")


class SensingDynamicsConvBlock(nn.Module):
    """2D convolution over EMG channels and time, with circular padding across channels.

    Args:
        in_channels: Input feature channels (1 for raw EMG).
        out_channels: Output feature channels.
        kernel_size: (kernel_C, kernel_T) — height (EMG channels) × width (time).
        stride: (stride_C, stride_T).
        dropout_rate: Dropout probability after activation.
        circular_pad: Whether to apply circular padding on the channel dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        dropout_rate: float = 0.05,
        circular_pad: bool = True,
    ):
        super().__init__()
        k_c, k_t = kernel_size

        self.circular_pad = CircularPad2d(k_c // 2) if circular_pad else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, k_t // 2),
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.smu = SMU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.circular_pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.smu(x)
        x = self.dropout(x)
        return x


class SensingDynamicsFeaturizer(nn.Module):
    """SensingDynamics featurizer: 2D convolutions over sEMG channels and time.

    Adapted from Sîmpetru et al. [2022a]. The original uses 3D convolutions
    (channels × patches × time) for a 320-electrode, 5-patch setup. This version
    uses 2D convolutions over (channels × time) for a single-patch device.

    The stack of ``SensingDynamicsConvBlock`` reduces both spatial dimensions, then
    adaptive average pooling collapses the channel dim to 1 so the featurizer works
    with any number of input EMG channels.
    """

    def __init__(
        self,
        conv_blocks: Sequence[SensingDynamicsConvBlock],
        out_channels: int,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.left_context = 0
        self.right_context = 0
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_emg, T) → (B, 1, C_emg, T) for 2D conv over (channels, time)
        x = x[:, None]  # (B, 1, C_emg, T)
        x = self.conv_blocks(x)  # (B, C_out, C_emg', T')
        # Adaptive pool channel dim to 1 — works with any number of EMG channels
        x = F.adaptive_avg_pool2d(x, (1, x.shape[-1]))  # (B, C_out, 1, T')
        x = x.squeeze(2)  # (B, C_out, T')
        return x  # (B, out_channels, T')
