# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from emg2pose.models.featurizers.tds import Conv1dBlock, SqueezeExcite


class CNNStage(nn.Module):
    """Stage with optional subsampling conv followed by stacked conv blocks."""

    def __init__(
        self,
        in_channels: int = 16,
        in_conv_kernel_width: int = 5,
        in_conv_stride: int = 1,
        num_blocks: int = 1,
        out_channels: int = 64,
        kernel_width: int = 3,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
        se: dict | None = None,
    ):
        super().__init__()
        layers: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()

        # Optional subsampling conv to desired channel width
        if in_conv_kernel_width > 0:
            layers["conv1dblock"] = Conv1dBlock(
                in_channels,
                out_channels,
                kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
                norm_type=norm_type,
                dropout=dropout,
                se=se if (se and se.get("enable", False)) else None,
            )
        elif in_channels != out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels ({out_channels}) "
                "if in_conv_kernel_width is not positive."
            )

        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            blocks.append(
                Conv1dBlock(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_width,
                    stride=1,
                    norm_type=norm_type,
                    dropout=dropout,
                    se=se if (se and se.get("enable", False)) else None,
                )
            )
        layers["conv_blocks"] = nn.Sequential(*blocks)

        self.layers = nn.Sequential(layers)
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.in_conv_kernel_width = in_conv_kernel_width
        self.in_conv_stride = in_conv_stride

    def forward(self, x):
        return self.layers(x)


class CNNFeaturizer(nn.Module):
    """
    Pure 1D CNN featurizer with the same config shape as TDS:
    - conv_blocks: list of Conv1dBlock
    - cnn_stages: list of CNNStage
    """

    def __init__(
        self,
        conv_blocks: Sequence[Conv1dBlock],
        cnn_stages: Sequence[CNNStage],
    ):
        super().__init__()
        self.layers = nn.Sequential(*conv_blocks, *cnn_stages)
        self.left_context = self._get_left_context(conv_blocks, cnn_stages)
        self.right_context = 0

    def forward(self, x):
        return self.layers(x)

    def _get_left_context(self, conv_blocks, cnn_stages) -> int:
        left, stride = 0, 1

        for conv_block in conv_blocks:
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

        for stage in cnn_stages:
            if stage.in_conv_kernel_width > 0:
                left += (stage.in_conv_kernel_width - 1) * stride
                stride *= stage.in_conv_stride

            left += (stage.kernel_width - 1) * stride * stage.layers["conv_blocks"].__len__()

        return left
