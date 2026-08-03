from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from egoemg.models.featurizers.tds import Conv1dBlock


class PatchCNNBlock(nn.Module):
    """Conv1d block with configurable activation and optional max pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: Literal["relu", "gelu"] = "gelu",
        pool_kernel_size: int | None = None,
        pool_stride: int | None = None,
    ) -> None:
        super().__init__()
        if activation == "relu":
            act: nn.Module = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            act,
        ]

        if pool_kernel_size is not None and pool_kernel_size > 0:
            layers.append(
                nn.MaxPool1d(
                    kernel_size=pool_kernel_size,
                    stride=pool_stride or pool_kernel_size,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchCNNFeaturizer(nn.Module):
    """Patch-wise CNN featurizer that operates independently per channel."""

    def __init__(
        self,
        patch_size: int,
        conv_blocks: Sequence[nn.Module],
        pad_mode: Literal["drop", "pad"] = "drop",
        pad_value: float = 0.0,
        pool_type: Literal["mean", "max", "last"] = "mean",
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if not conv_blocks:
            raise ValueError("conv_blocks must be a non-empty sequence.")

        if isinstance(conv_blocks[0], Conv1dBlock):
            first_in = conv_blocks[0].conv[0].in_channels
        elif isinstance(conv_blocks[0], PatchCNNBlock):
            first_in = conv_blocks[0].net[0].in_channels
        else:
            raise ValueError(
                "conv_blocks must contain Conv1dBlock or PatchCNNBlock instances."
            )

        if isinstance(conv_blocks[-1], Conv1dBlock):
            last_out = conv_blocks[-1].conv[0].out_channels
        elif isinstance(conv_blocks[-1], PatchCNNBlock):
            last_out = conv_blocks[-1].net[0].out_channels
        else:
            raise ValueError(
                "conv_blocks must contain Conv1dBlock or PatchCNNBlock instances."
            )

        if first_in != 1:
            raise ValueError(
                "conv_blocks[0] must accept 1 input channel for per-channel patches."
            )

        self.patch_size = patch_size
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.pool_type = pool_type
        self.cnn = nn.Sequential(*conv_blocks)
        self.out_channels = last_out

        # First output depends on a full patch.
        self.left_context = patch_size - 1
        self.right_context = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        bsz, channels, time = x.shape

        if self.pad_mode == "pad":
            pad_len = (-time) % self.patch_size
            if pad_len:
                x = F.pad(x, (0, pad_len), value=self.pad_value)
                time = x.shape[-1]
        elif self.pad_mode == "drop":
            time = (time // self.patch_size) * self.patch_size
            if time == 0:
                raise ValueError("Input too short for patch_size.")
            x = x[..., :time]
        else:
            raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")

        num_patches = time // self.patch_size
        if num_patches == 0:
            raise ValueError("No patches could be formed from the input.")

        patches = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        patches = patches.contiguous().view(bsz * channels * num_patches, 1, -1)

        features = self.cnn(patches)  # (B*C*num_patches, out_channels, T_patch)
        if self.pool_type == "mean":
            pooled = features.mean(dim=-1)
        elif self.pool_type == "max":
            pooled = features.amax(dim=-1)
        elif self.pool_type == "last":
            pooled = features[..., -1]
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")

        pooled = pooled.view(bsz, channels, num_patches, self.out_channels)
        return pooled.permute(0, 1, 3, 2).reshape(
            bsz, channels * self.out_channels, num_patches
        )
