import collections
from collections.abc import Sequence

from typing import Literal

import torch

from torch import nn


class SqueezeExcite(nn.Module):
    """Channel-wise Squeeze-Excitation over temporal dimension."""
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        residual: bool = True,
        mode: Literal["causal", "global"] = "causal",
    ):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.residual = residual
        self.mode = mode
        # 1x1 Conv1d <=> per-time-step Linear over channels
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape

        if self.mode == "causal":
            # prefix mean: scale_pool[..., t] = mean_{k<=t} x[..., k]
            cumsum = torch.cumsum(x, dim=-1)  # (B, C, T)
            denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(
                1, 1, T
            )
            scale_pool = cumsum / denom  # (B, C, T) strictly causal
        else:
            # global mean over time, then broadcast
            scale_pool = x.mean(dim=-1, keepdim=True).expand(-1, -1, T)

        scale = self.net(scale_pool)  # (B, C, T)
        out = x * scale
        return out + x if self.residual else out

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
        padding: int = 0,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
        se: dict | None = None,
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
            padding=padding,
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
        self.se = (
            None
            if not se
            else SqueezeExcite(
                channels=out_channels,
                reduction=int(se.get("reduction", 4)),
                residual=bool(se.get("residual", True)),
                mode=str(se.get("mode", "causal")),
            )
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type == "layer":
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        if self.se is not None:
            x = self.se(x)
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

    def __init__(
        self,
        channels: int,
        width: int,
        kernel_width: int,
        time_padding: int = 0,
        se: dict | None = None,
    ) -> None:
        super().__init__()

        assert kernel_width % 2, "kernel_width must be odd."
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            dilation=(1, 1),
            stride=(1, 1),
            padding=(0, time_padding),
            groups=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)

        self.channels = channels
        self.width = width
        self.se = (
            None
            if not se
            else SqueezeExcite(
                channels=channels * width,
                reduction=int(se.get("reduction", 4)),
                residual=bool(se.get("residual", True)),
                mode=str(se.get("mode", "causal")),
            )
        )

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

        if self.se is not None:
            x = self.se(x)
        return x


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int, se: dict | None = None) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)
        self.se = (
            None
            if not se
            else SqueezeExcite(
                channels=num_features,
                reduction=int(se.get("reduction", 4)),
                residual=bool(se.get("residual", True)),
                mode=str(se.get("mode", "causal")),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = inputs
        x = x.swapaxes(-1, -2)  # BCT -> BTC
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)  # BTC -> BCT
        x += inputs

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        if self.se is not None:
            x = self.se(x)
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
        time_padding: int = 0,
        se: dict | None = None,
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
                    TDSConv2dBlock(
                        channels,
                        feature_width,
                        kernel_width,
                        time_padding=time_padding,
                        se=se,
                    ),
                    TDSFullyConnectedBlock(num_features, se=se),
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
        in_conv_padding: int = 0,
        num_blocks: int = 1,
        channels: int = 8,
        feature_width: int = 2,
        kernel_width: int = 1,
        kernel_time_padding: int = 0,
        out_channels: int | None = None,
        se: dict | None = None,
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
                padding=in_conv_padding,
                se=se if (se and se.get("enable", False)) else None,
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
            time_padding=kernel_time_padding,
            se=se if (se and se.get("enable", False)) else None,
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
        self,
        conv_blocks: Sequence[Conv1dBlock],
        tds_stages: Sequence[TdsStage],
        se: dict | None = None,
    ):
        super().__init__()
        if se and se.get("enable", False):
            for block in conv_blocks:
                block.se = SqueezeExcite(
                    channels=block.conv[0].out_channels,
                    reduction=int(se.get("reduction", 4)),
                    residual=bool(se.get("residual", True)),
                    mode=str(se.get("mode", "causal")),
                )
            for stage in tds_stages:
                if hasattr(stage, "layers"):
                    conv = stage.layers[0] if len(stage.layers) > 0 else None
                    if conv is not None and isinstance(conv, Conv1dBlock):
                        conv.se = SqueezeExcite(
                            channels=conv.conv[0].out_channels,
                            reduction=int(se.get("reduction", 4)),
                            residual=bool(se.get("residual", True)),
                            mode=str(se.get("mode", "causal")),
                        )
                if hasattr(stage, "layers"):
                    for layer in stage.layers:
                        if isinstance(layer, TDSConvEncoder):
                            for sub in layer.tds_conv_blocks:
                                if isinstance(sub, (TDSConv2dBlock, TDSFullyConnectedBlock)):
                                    sub.se = SqueezeExcite(
                                        channels=sub.layer_norm.normalized_shape[0],
                                        reduction=int(se.get("reduction", 4)),
                                        residual=bool(se.get("residual", True)),
                                        mode=str(se.get("mode", "causal")),
                                    )
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
