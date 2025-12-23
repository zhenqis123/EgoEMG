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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock1dT(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention1d(nn.Module):
    def __init__(self, channels: int, num_heads: int | None = None) -> None:
        super().__init__()
        if num_heads is None:
            num_heads = 4 if channels % 4 == 0 and channels >= 32 else 1
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = h.transpose(1, 2)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        return x + attn_out.transpose(1, 2)


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


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        cond_channels: int = 16,
        channels: Sequence[int] = (32, 64, 128, 256),
        time_dim: int = 256,
        attn_heads: int | None = None,
        cond_mode: str = "concat",
    ) -> None:
        super().__init__()
        if cond_mode not in {"concat"}:
            raise ValueError(f"Unsupported cond_mode: {cond_mode}")
        self.cond_mode = cond_mode
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        input_channels = in_channels + cond_channels
        self.input_proj = nn.Conv1d(input_channels, channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.attn_downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_in = channels[0]
        for ch_out in channels:
            self.downs.append(ResBlock1dT(ch_in, ch_out, time_dim))
            self.attn_downs.append(SelfAttention1d(ch_out, num_heads=attn_heads))
            self.downsamples.append(Downsample1d(ch_out))
            ch_in = ch_out

        self.mid_block1 = ResBlock1dT(ch_in, ch_in, time_dim)
        self.mid_attn = SelfAttention1d(ch_in, num_heads=attn_heads)
        self.mid_block2 = ResBlock1dT(ch_in, ch_in, time_dim)

        self.ups = nn.ModuleList()
        self.attn_ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for ch_out in reversed(channels):
            self.upsamples.append(Upsample1d(ch_in))
            self.ups.append(ResBlock1dT(ch_in + ch_out, ch_out, time_dim))
            self.attn_ups.append(SelfAttention1d(ch_out, num_heads=attn_heads))
            ch_in = ch_out

        self.out_norm = nn.GroupNorm(_num_groups(ch_in), ch_in)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(ch_in, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.cond_mode == "concat":
            x = torch.cat([x, cond], dim=1)
        t_emb = self.time_embed(t)

        h = self.input_proj(x)
        skips = []
        for res, attn, down in zip(self.downs, self.attn_downs, self.downsamples, strict=True):
            h = res(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for up, res, attn, skip in zip(
            self.upsamples, self.ups, self.attn_ups, reversed(skips), strict=True
        ):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = res(h, t_emb)
            h = attn(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h
