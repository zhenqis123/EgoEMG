# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import collections
from collections.abc import Sequence

from typing import Literal

import torch

from torch import nn


##################
# TDS FEATURIZER #
##################


class Permute(nn.Module):
    """Permute the dimensions of the input tensor.
    For example:
    ```
    Permute('NTC', 'NCT') == x.permute(0, 2, 1)
    ```
    """

    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        assert len(from_dims) == len(
            to_dims
        ), "Same number of from- and to- dimensions should be specified for"

        if len(from_dims) not in {3, 4, 5, 6}:
            raise ValueError(
                "Only 3, 4, 5, and 6D tensors supported in Permute for now"
            )

        self.from_dims = from_dims
        self.to_dims = to_dims
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]

    def get_inverse_permute(self) -> "Permute":
        "Get the permute operation to get us back to the original dim order"
        return Permute(from_dims=self.to_dims, to_dims=self.from_dims)

    def __repr__(self):
        return f"Permute({self.from_dims!r} => {self.to_dims!r})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)


class BatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d except in NTC format"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.permute_forward = Permute("NTC", "NCT")
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        self.permute_back = Permute("NCT", "NTC")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.permute_back(self.bn(self.permute_forward(inputs)))


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
    ):
        """A 1D convolution with padding so the input and output lengths match."""

        super().__init__()

        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

        layers = {}
        layers["conv1d"] = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        if norm_type == "batch":
            layers["norm"] = BatchNorm1d(out_channels)

        layers["relu"] = nn.ReLU(inplace=True)
        layers["dropout"] = nn.Dropout(dropout)

        self.conv = nn.Sequential(
            *[layers[key] for key in layers if layers[key] is not None]
        )

        if norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type == "layer":
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()

        assert kernel_width % 2, "kernel_width must be odd."
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            dilation=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)

        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        B, C, T = inputs.shape  # BCT

        # BCT -> BcwT
        x = inputs.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(B, C, -1)  # BcwT -> BCT

        # Skip connection after downsampling
        T_out = x.shape[-1]
        x = x + inputs[..., -T_out:]

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = inputs
        x = x.swapaxes(-1, -2)  # BCT -> BTC
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)  # BTC -> BCT
        x += inputs

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)

        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            assert (
                num_features % channels == 0
            ), f"block_channels {channels} must evenly divide num_features {num_features}"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, feature_width, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class TdsStage(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        in_conv_kernel_width: int = 5,
        in_conv_stride: int = 1,
        num_blocks: int = 1,
        channels: int = 8,
        feature_width: int = 2,
        kernel_width: int = 1,
        out_channels: int | None = None,
    ):
        super().__init__()
        """Stage of several TdsBlocks preceded by a non-separable sub-sampling conv.

        The initial (and optionally sub-sampling) conv layer maps the number of
        input channels to the corresponding internal width used by the residual TDS
        blocks.

        Follows the multi-stage network construction from
        https://arxiv.org/abs/1904.02619.
        """

        layers: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()

        C = channels * feature_width

        self.out_channels = out_channels

        # Conv1d block
        if in_conv_kernel_width > 0:
            layers["conv1dblock"] = Conv1dBlock(
                in_channels,
                C,
                kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
            )
        elif in_channels != C:
            # Check that in_channels is consistent with TDS
            # channels and feature width
            raise ValueError(
                f"in_channels ({in_channels}) must equal channels *"
                f" feature_width ({channels} * {feature_width}) if"
                " in_conv_kernel_width is not positive."
            )

        # TDS block
        layers["tds_block"] = TDSConvEncoder(
            num_features=C,
            block_channels=[channels] * num_blocks,
            kernel_width=kernel_width,
        )

        # Linear projection
        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = self.layers(x)
        if self.out_channels is not None:
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TdsNetwork(nn.Module):
    def __init__(
        self, conv_blocks: Sequence[Conv1dBlock], tds_stages: Sequence[TdsStage]
    ):
        super().__init__()
        self.layers = nn.Sequential(*conv_blocks, *tds_stages)
        self.left_context = self._get_left_context(conv_blocks, tds_stages)
        self.right_context = 0

    def forward(self, x):
        return self.layers(x)

    def _get_left_context(self, conv_blocks, tds_stages) -> int:
        left, stride = 0, 1

        for conv_block in conv_blocks:
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

        for tds_stage in tds_stages:

            conv_block = tds_stage.layers.conv1dblock
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

            tds_block = tds_stage.layers.tds_block
            for _ in range(tds_block.num_blocks):
                left += (tds_block.kernel_width - 1) * stride

        return left


#############
# NEUROPOSE #
#############


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
            """Single convolution block."""
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]

        modules = [*_conv(in_channels, out_channels)]
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
        encoder_blocks: list[EncoderBlock],
        residual_blocks: list[ResidualBlock],
        decoder_blocks: list[DecoderBlock],
        linear_in_channels: int,
        out_channels: int = 22,
    ):
        super().__init__()
        self.network = nn.Sequential(*encoder_blocks, *residual_blocks, *decoder_blocks)
        self.linear = nn.Linear(linear_in_channels, out_channels)
        self.left_context = 0
        self.right_context = 0

    def forward(self, x):
        # Neuropose uses 2D convolutions over time and space, so we add a new
        # channel dimension corresponding to the network features.
        x = x[:, None].swapaxes(-1, -2)  # BCT -> BCtc
        x = self.network(x)
        x = x.swapaxes(-2, -3).flatten(-2)  # BCtc -> BTC
        return self.linear(x).swapaxes(-1, -2)  # BTC -> BCT


############
# DECODERS #
############


class MLP(nn.Module):
    """Basic MLP with optional scaling of the final output."""

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
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if layer_norm:
                layers.append(nn.LayerNorm(out_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(sizes[-1], out_channels))

        self.mlp = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x):
        # x is (batch, channel)
        return self.mlp(x) * self.scale


class SequentialLSTM(nn.Module):
    """
    LSTM where each forward() call computes only a single time step, to be compatible
    looping over time manually.

    NOTE: Need to manually reset the state in outer context after each trajectory!
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_layers: int = 1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True)
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.mlp_out = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(hidden_size, out_channels)
        )
        self.scale = scale

    def reset_state(self):
        self.hidden = None

    def forward(self, x):
        """Forward pass for a single time step, where x is (batch, channel.)"""

        if self.hidden is None:
            # Initialize hidden state with zeros
            batch_size = x.size(0)
            device = x.device
            size = (self.num_layers, batch_size, self.hidden_size)
            self.hidden = (torch.zeros(*size).to(device), torch.zeros(*size).to(device))

        out, self.hidden = self.lstm(x[:, None], self.hidden)
        return self.mlp_out(out[:, 0]) * self.scale

    def _non_sequential_forward(self, x):
        """Non-sequential forward pass, where x is (batch, time, channel)."""
        return self.mlp_out(self.lstm(x)[0]) * self.scale
