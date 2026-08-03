# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        max_pool_size: tuple[int, int],
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)
        maxpool = nn.MaxPool2d(kernel_size=max_pool_size, stride=max_pool_size)
        self.network = nn.Sequential(conv, bn, relu, dropout, maxpool)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        num_convs: int,
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        def _conv(in_channels: int, out_channels: int):
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]

        modules: list[nn.Module] = [*_conv(in_channels, out_channels)]
        for _ in range(num_convs - 1):
            modules += _conv(out_channels, out_channels)

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.network(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        upsampling: tuple[int, int],
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)
        scale_factor = (float(upsampling[0]), float(upsampling[1]))
        upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.network = nn.Sequential(conv, bn, relu, dropout, upsample)
        self.out_channels = out_channels

    def forward(self, x):
        return self.network(x)


class NeuroPose(nn.Module):
    def __init__(
        self,
        encoder_blocks: Sequence[EncoderBlock],
        residual_blocks: Sequence[ResidualBlock],
        decoder_blocks: Sequence[DecoderBlock],
        linear_in_channels: int,
        out_channels: int = 22,
    ):
        super().__init__()
        self.network = nn.Sequential(*encoder_blocks, *residual_blocks, *decoder_blocks)
        self.linear = nn.Linear(linear_in_channels, out_channels)
        self.left_context = 0
        self.right_context = 0

    def forward(self, x):
        x = x[:, None].swapaxes(-1, -2)  # BCT -> BCtc
        x = self.network(x)
        x = x.swapaxes(-2, -3).flatten(-2)  # BCtc -> BTC
        return self.linear(x).swapaxes(-1, -2)  # BTC -> BCT
