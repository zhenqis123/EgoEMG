from typing import Literal

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return x + self.pe[: x.shape[1], :].unsqueeze(0)


class TransformerDecoder(nn.Module):
    """Configurable Transformer (encoder stack) operating on features (B, C, T)."""

    def __init__(
        self,
        in_channels: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        norm_first: bool = True,
        causal: bool = False,
        pos_encoding: Literal["none", "sinusoidal"] = "none",
        out_proj: bool = True,
        preset: str | None = None,  # absorbs preset overrides from config defaults
    ):
        super().__init__()
        self.causal = causal
        self.input_proj = (
            nn.Identity() if in_channels == model_dim else nn.Linear(in_channels, model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_encoder = (
            SinusoidalPositionalEncoding(model_dim) if pos_encoding == "sinusoidal" else None
        )
        self.output_proj = nn.Linear(model_dim, model_dim) if out_proj else nn.Identity()

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(T, T, device=device, dtype=torch.bool).triu(1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, T).
        Returns:
            Tensor of shape (B, C_out=model_dim, T).
        """
        tgt = x.transpose(1, 2)  # (B, T, C)
        tgt = self.input_proj(tgt)

        if self.pos_encoder is not None:
            tgt = self.pos_encoder(tgt)

        src_mask = None
        if self.causal:
            src_mask = self._build_causal_mask(tgt.shape[1], tgt.device)

        encoded = self.encoder(src=tgt, mask=src_mask)
        decoded = self.output_proj(encoded)
        return decoded.transpose(1, 2)
