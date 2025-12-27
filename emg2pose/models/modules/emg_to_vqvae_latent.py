from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class EmgToVQVAELatent(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        conv_channels: Sequence[int] = (64, 128, 128, 128),
        dropout: float = 0.2,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        num_codes: int = 128,
        num_groups: int = 5,
        align_kernel: int = 31,
        conv_kernels: Sequence[int] | None = None,
        conv_strides: Sequence[int] | None = None,
        conv_paddings: Sequence[int] | None = None,
        output_steps: int | None = None,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 4:
            raise ValueError("Expected 4 conv channels for the 4 Conv1d layers.")

        kernels = tuple(conv_kernels) if conv_kernels is not None else (10, 6, 3, 3)
        strides = tuple(conv_strides) if conv_strides is not None else (5, 2, 1, 1)
        paddings = (
            tuple(conv_paddings) if conv_paddings is not None else (4, 2, 1, 1)
        )
        if not (len(kernels) == len(strides) == len(paddings) == len(conv_channels)):
            raise ValueError(
                "conv_kernels/strides/paddings must match conv_channels length."
            )

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out, k, s, p in zip(
            conv_channels, kernels, strides, paddings, strict=True
        ):
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p),
                    nn.BatchNorm1d(c_out),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            c_in = c_out
        self.encoder = nn.Sequential(*layers)

        self.temporal = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        align_channels = 2 * gru_hidden
        self.alignment = nn.Conv1d(
            align_channels,
            align_channels,
            kernel_size=align_kernel,
            padding=align_kernel // 2,
            groups=align_channels,
            bias=True,
        )
        self._init_alignment_identity()

        self.proj = nn.Linear(align_channels, num_groups * num_codes)
        self.num_codes = int(num_codes)
        self.num_groups = int(num_groups)
        self.output_steps = output_steps

    def _init_alignment_identity(self) -> None:
        with torch.no_grad():
            self.alignment.weight.zero_()
            center = self.alignment.weight.shape[-1] // 2
            self.alignment.weight[:, 0, center] = 1.0
            if self.alignment.bias is not None:
                self.alignment.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have shape (B, C, T), got {tuple(x.shape)}"
            )
        b, _, _ = x.shape

        feats = self.encoder(x)
        feats_t = feats.transpose(1, 2)  # (B, 200, C)
        feats_t, _ = self.temporal(feats_t)  # (B, 200, 2H)
        feats = feats_t.transpose(1, 2)  # (B, 2H, 200)
        feats = self.alignment(feats)  # (B, 2H, 200)
        if self.output_steps is not None and feats.shape[-1] != self.output_steps:
            feats = F.interpolate(
                feats, size=self.output_steps, mode="linear", align_corners=True
            )
        feats_t = feats.transpose(1, 2)  # (B, T, 2H)

        proj = self.proj(feats_t)  # (B, 200, 5*K)
        proj = proj.view(b, -1, self.num_groups, self.num_codes)  # (B, 200, 5, K)
        proj = proj.permute(0, 2, 1, 3).contiguous()  # (B, 5, 200, K)
        logits = proj.view(b, self.num_groups * proj.shape[2], self.num_codes)
        return logits
