# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T


def _to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


def _conv2d_out_size(
    size: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class SequenceMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_sizes: Sequence[int] | Iterable[int] = (256,),
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        act: nn.Module = nn.ReLU() if activation == "relu" else nn.GELU()

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
        # x: (B, T, C)
        b, t, c = x.shape
        x = x.reshape(b * t, c)
        out = self.net(x)
        return out.view(b, t, -1)


class EmgConformer(nn.Module):
    def __init__(
        self,
        out_channels: int = 20,
        spectrogram: dict | None = None,
        masking: dict | None = None,
        spatial_dropout: float = 0.0,
        conv2d_layers: Sequence[dict] | None = None,
        conformer: dict | None = None,
        head: dict | None = None,
        log_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        spectrogram = spectrogram or {}
        masking = masking or {}
        conformer = conformer or {}
        head = head or {}
        conv2d_layers = conv2d_layers or []

        self.out_channels = out_channels
        self.log_eps = float(log_eps)
        self.left_context = 0
        self.right_context = 0

        self.base_spectrogram = T.Spectrogram(
            n_fft=int(spectrogram.get("n_fft", 64)),
            win_length=int(spectrogram.get("win_length", 64)),
            hop_length=int(spectrogram.get("hop_length", 16)),
            power=float(spectrogram.get("power", 2.0)),
            center=bool(spectrogram.get("center", True)),
        )

        aggregation_mask = torch.zeros(6, 33)
        intervals = [(1, 2), (2, 4), (4, 8), (8, 12), (12, 22), (22, 33)]
        for i, (start, end) in enumerate(intervals):
            aggregation_mask[i, start:end] = 1.0
        self.register_buffer("rsg_matrix", aggregation_mask)

        self.spec_augment = T.SpecAugment(
            n_time_masks=int(masking.get("n_time_masks", 0)),
            time_mask_param=int(masking.get("time_mask_param", 0)),
            n_freq_masks=int(masking.get("n_freq_masks", 0)),
            freq_mask_param=int(masking.get("freq_mask_param", 0)),
            iid_masks=bool(masking.get("iid_masks", True)),
            p=float(masking.get("p", 1.0)),
            zero_masking=bool(masking.get("zero_masking", False)),
        )

        self.spatial_dropout = (
            nn.Dropout2d(p=float(spatial_dropout))
            if spatial_dropout > 0
            else nn.Identity()
        )

        conv_layers: list[nn.Module] = []
        if conv2d_layers and "in_channels" not in conv2d_layers[0]:
            raise ValueError(
                "conv2d_layers[0].in_channels is required for EmgConformer."
            )
        if conv2d_layers:
            in_channels = int(conv2d_layers[0].get("in_channels", 1))
        else:
            in_channels = 1
        freq_bins = 6
        for layer in conv2d_layers:
            out_ch = int(layer["out_channels"])
            kernel = _to_2tuple(layer.get("kernel_size", 3))
            stride = _to_2tuple(layer.get("stride", 1))
            padding = _to_2tuple(layer.get("padding", 0))
            dilation = _to_2tuple(layer.get("dilation", 1))
            conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=not layer.get("norm", False),
                )
            )
            if layer.get("norm", False):
                conv_layers.append(nn.BatchNorm2d(out_ch))
            act = layer.get("activation", "relu")
            conv_layers.append(nn.ReLU() if act == "relu" else nn.GELU())
            if layer.get("dropout", 0.0) > 0:
                conv_layers.append(nn.Dropout2d(float(layer["dropout"])))
            in_channels = out_ch
            freq_bins = _conv2d_out_size(
                freq_bins, kernel[0], stride[0], padding[0], dilation[0]
            )

        self.conv = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()
        conformer_input_dim = int(conformer.get("input_dim", in_channels * freq_bins))
        self.conformer = torchaudio.models.Conformer(
            input_dim=conformer_input_dim,
            num_heads=int(conformer.get("num_heads", 4)),
            ffn_dim=int(conformer.get("ffn_dim", 256)),
            num_layers=int(conformer.get("num_layers", 4)),
            depthwise_conv_kernel_size=int(
                conformer.get("depthwise_conv_kernel_size", 31)
            ),
            dropout=float(conformer.get("dropout", 0.1)),
        )

        self.head = SequenceMLP(
            in_channels=conformer_input_dim,
            out_channels=out_channels,
            hidden_sizes=head.get("hidden_sizes", (256,)),
            activation=head.get("activation", "relu"),
            dropout=float(head.get("dropout", 0.0)),
        )

    def _apply_masking(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return self.spec_augment(x)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        # emg: (B, C, T)
        linear_spec = self.base_spectrogram(emg)
        log_spec = torch.log10(linear_spec + self.log_eps)
        rsg_features = torch.einsum("kf,bcft->bckt", self.rsg_matrix, log_spec)
        rsg_features = self._apply_masking(rsg_features)
        rsg_features = self.spatial_dropout(rsg_features)

        x = self.conv(rsg_features)
        if x.ndim != 4:
            raise RuntimeError("Expected conv output to be 4D (B, C, F, T).")
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)
        x = self.head(x)
        return x.transpose(1, 2).contiguous()
