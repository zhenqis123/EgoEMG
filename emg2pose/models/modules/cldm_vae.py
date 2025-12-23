from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F


def _num_groups(channels: int) -> int:
    max_groups = min(8, channels)
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


def _sinusoidal_position_encoding(
    length: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if length <= 0:
        raise ValueError("length must be positive.")
    half = dim // 2
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    div_term = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / max(half, 1))
    )[None, :]
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    if half > 0:
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
    if dim % 2 == 1:
        pe[:, -1] = 0.0
    return pe.unsqueeze(0)


class ResBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention1d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int | None = None,
        num_groups: int | None = None,
        expected_length: int | None = None,
        attn_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_heads is None:
            num_heads = 1
        if num_groups is not None and num_groups <= 0:
            raise ValueError("num_groups must be positive or None.")
        self.channels = channels
        self.num_groups = num_groups
        self.expected_length = expected_length
        self.norm = nn.GroupNorm(_num_groups(channels), channels)

        if self.num_groups is None:
            embed_dim_in = channels
        else:
            if self.expected_length is None:
                raise ValueError("expected_length is required when num_groups is set.")
            if self.expected_length % self.num_groups != 0:
                raise ValueError("expected_length must be divisible by num_groups.")
            patch_size = self.expected_length // self.num_groups
            embed_dim_in = channels * patch_size
            self.patch_size = patch_size
        if attn_embed_dim is None:
            attn_embed_dim = embed_dim_in
        if attn_embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.proj_in = (
            nn.Linear(embed_dim_in, attn_embed_dim)
            if attn_embed_dim != embed_dim_in
            else nn.Identity()
        )
        self.proj_out = (
            nn.Linear(attn_embed_dim, embed_dim_in)
            if attn_embed_dim != embed_dim_in
            else nn.Identity()
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        b, c, t = h.shape
        if self.num_groups is None:
            h = h.transpose(1, 2)  # BCT -> BTC
            h = self.proj_in(h)
            h = h + _sinusoidal_position_encoding(
                h.shape[1], h.shape[2], device=h.device, dtype=h.dtype
            )
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            attn_out = self.proj_out(attn_out)
            return x + attn_out.transpose(1, 2)

        expected = self.expected_length or t
        if t < expected:
            h = F.pad(h, (0, expected - t))
        elif t > expected:
            h = h[..., :expected]
        t_eff = h.shape[-1]
        patch_size = t_eff // self.num_groups
        if t_eff % self.num_groups != 0:
            raise ValueError("time length must be divisible by num_groups.")

        h = h.view(b, c, self.num_groups, patch_size)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = h.view(b, self.num_groups, c * patch_size)
        h = self.proj_in(h)
        h = h + _sinusoidal_position_encoding(
            h.shape[1], h.shape[2], device=h.device, dtype=h.dtype
        )

        attn_out, _ = self.attn(h, h, h, need_weights=False)
        attn_out = self.proj_out(attn_out)
        attn_out = attn_out.view(b, self.num_groups, patch_size, c)
        attn_out = attn_out.permute(0, 3, 1, 2).contiguous()
        attn_out = attn_out.view(b, c, t_eff)
        if t_eff != t:
            attn_out = attn_out[..., :t]
        return x + attn_out


class Downsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class VAE1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (32, 64, 64, 64),
        latent_channels: int = 16,
        attn_heads: int | None = None,
        attn_group_counts: Sequence[int] | None = None,
        input_length: int = 4000,
        attn_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.latent_channels = int(latent_channels)

        if attn_group_counts is None:
            attn_group_counts = [200, 100, 50, 25]
        if len(attn_group_counts) != len(channels):
            raise ValueError(
                "attn_group_counts must match number of layers in channels."
            )
        layer_lengths = [input_length // (2**i) for i in range(len(channels))]
        if layer_lengths[-1] <= 0:
            raise ValueError("input_length too small for the number of layers.")
        enc_layers: list[nn.Module] = []
        ch_in = self.in_channels
        for idx, (ch_out, groups) in enumerate(
            zip(channels, attn_group_counts, strict=True)
        ):
            enc_layers.extend(
                [
                    ResBlock1d(ch_in, ch_out),
                    SelfAttention1d(
                        ch_out,
                        num_heads=attn_heads,
                        num_groups=groups,
                        expected_length=layer_lengths[idx],
                        attn_embed_dim=attn_embed_dim,
                    ),
                    Downsample1d(ch_out),
                ]
            )
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_layers)
        self.to_mu = nn.Conv1d(ch_in, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv1d(ch_in, latent_channels, kernel_size=1)

        dec_layers: list[nn.Module] = []
        ch_in = latent_channels
        self.from_latent = nn.Conv1d(latent_channels, channels[-1], kernel_size=1)
        ch_in = channels[-1]
        dec_groups = list(reversed(attn_group_counts))
        dec_lengths = list(reversed(layer_lengths))
        for ch_out, groups, length in zip(
            reversed(channels), dec_groups, dec_lengths, strict=True
        ):
            dec_layers.extend(
                [
                    Upsample1d(ch_in),
                    ResBlock1d(ch_in, ch_out),
                    SelfAttention1d(
                        ch_out,
                        num_heads=attn_heads,
                        num_groups=groups,
                        expected_length=length,
                        attn_embed_dim=attn_embed_dim,
                    ),
                ]
            )
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec_layers)
        self.out_conv = nn.Conv1d(ch_in, out_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.decoder(h)
        return self.out_conv(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)
