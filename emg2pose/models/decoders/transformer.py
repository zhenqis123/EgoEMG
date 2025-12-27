from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


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


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding helper."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even dim.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: (B, H, T, D)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2)


class RotaryMultiheadAttention(nn.Module):
    """Multihead attention with rotary positional embeddings."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("Rotary attention requires an even head_dim.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = float(dropout)
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.rope = RotaryEmbedding(head_dim, base=rope_base)
        self.batch_first = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if key is not query or value is not query:
            raise ValueError("RotaryMultiheadAttention expects q=k=v for encoder.")
        b, t, _ = query.shape

        qkv = self.in_proj(query)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(t, device=query.device, dtype=query.dtype)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        combined_mask = attn_mask
        if key_padding_mask is not None:
            key_mask = key_padding_mask.view(b, 1, 1, t)
            if combined_mask is None:
                combined_mask = key_mask
            elif combined_mask.dtype == torch.bool:
                if combined_mask.dim() == 2:
                    combined_mask = combined_mask[None, None, :, :]
                combined_mask = combined_mask | key_mask
            else:
                mask_val = torch.finfo(combined_mask.dtype).min
                combined_mask = combined_mask + key_mask.to(combined_mask.dtype) * mask_val

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = attn.transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        out = self.out_proj(out)
        if need_weights:
            raise NotImplementedError("RotaryMultiheadAttention does not return weights.")
        return out, None


class RotaryTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with rotary positional embeddings."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = RotaryMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def _sa_block(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        out, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.dropout1(out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + self._ff_block(self.norm2(src))
            return src
        src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
        src = self.norm2(src + self._ff_block(src))
        return src


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
        pos_encoding: Literal["none", "sinusoidal", "rope"] = "none",
        out_proj: bool = True,
        preset: str | None = None,  # absorbs preset overrides from config defaults
    ):
        super().__init__()
        self.causal = causal
        self.input_proj = (
            nn.Identity() if in_channels == model_dim else nn.Linear(in_channels, model_dim)
        )
        if pos_encoding == "rope":
            layer = RotaryTransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            )
        else:
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
