from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from emg2pose.models.decoders.transformer import TransformerDecoder


class EmgToVQVAETransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        norm_first: bool = True,
        causal: bool = False,
        pos_encoding: Literal["none", "sinusoidal", "rope"] = "sinusoidal",
        input_kernel: int = 1,
        input_stride: int = 1,
        input_padding: int | None = None,
        num_codes: int = 128,
        num_groups: int = 5,
        output_steps: int | None = None,
    ) -> None:
        super().__init__()
        if input_padding is None:
            input_padding = input_kernel // 2 if input_kernel > 1 else 0

        self.input_proj = nn.Conv1d(
            in_channels,
            model_dim,
            kernel_size=input_kernel,
            stride=input_stride,
            padding=input_padding,
        )
        self.transformer = TransformerDecoder(
            in_channels=model_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            causal=causal,
            pos_encoding=pos_encoding,
            out_proj=False,
        )
        self.proj = nn.Linear(model_dim, num_groups * num_codes)
        self.num_codes = int(num_codes)
        self.num_groups = int(num_groups)
        self.output_steps = output_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have shape (B, C, T), got {tuple(x.shape)}"
            )
        b, _, _ = x.shape

        feats = self.input_proj(x)
        feats = self.transformer(feats)
        if self.output_steps is not None and feats.shape[-1] != self.output_steps:
            feats = F.interpolate(
                feats, size=self.output_steps, mode="linear", align_corners=True
            )
        feats_t = feats.transpose(1, 2)
        proj = self.proj(feats_t)
        proj = proj.view(b, -1, self.num_groups, self.num_codes)
        proj = proj.permute(0, 2, 1, 3).contiguous()
        logits = proj.view(b, self.num_groups * proj.shape[2], self.num_codes)
        return logits
