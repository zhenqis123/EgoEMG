"""
PhysioWave temporal regression model for per-timestep pose outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from emg2pose.models.physiowave.head_modules import (
    RegressionHead,
    TemporalRegressionHead,
)
from emg2pose.models.physiowave.transformer_modules import (
    PatchEmbed,
    PositionEmbedding,
    TransformerEncoder,
)
from emg2pose.models.physiowave.wavelet_modules import SoftGateWaveletDecomp


def _as_tuple(value: int | Iterable[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be int or tuple/list of length 2.")


def load_pretrained_feature_extractor(
    model: nn.Module,
    pretrained_path: str,
    strict: bool = False,
) -> None:
    """Load pretrained weights with shape filtering."""
    path = Path(pretrained_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            pretrained_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint
    else:
        pretrained_dict = checkpoint

    model_dict = model.state_dict()
    filtered_dict = {}
    for key, value in pretrained_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            filtered_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
    if strict:
        allow_missing_prefixes = ("input_adapter.", "head.")
        filtered_missing = [
            key for key in missing_keys if not key.startswith(allow_missing_prefixes)
        ]
        if filtered_missing or unexpected_keys:
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{model.__class__.__name__}:\n"
                f"\tMissing key(s) in state_dict: {filtered_missing}\n"
                f"\tUnexpected key(s) in state_dict: {unexpected_keys}\n"
            )


class PhysioWaveTemporalRegressor(nn.Module):
    """
    PhysioWave backbone with temporal regression head.
    """

    def __init__(
        self,
        input_channels: int | None = None,
        in_channels: int = 8,
        max_level: int = 3,
        wave_kernel_size: int = 16,
        wavelet_names: list[str] | None = None,
        use_separate_channel: bool = True,
        patch_size: int | tuple[int, int] = 64,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_pos_embed: bool = True,
        pos_embed_type: str = "2d",
        temporal_pooling: str = "mean",
        upsample_mode: str = "linear",
        regression_mode: str = "temporal",
        out_channels: int = 20,
        head_hidden_dim: int | None = 512,
        head_dropout: float = 0.1,
        freeze_pretrained: bool = False,
        pretrained_path: str | None = None,
        pretrained_strict: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.max_level = max_level
        self.patch_size = _as_tuple(patch_size, "patch_size")
        self.embed_dim = embed_dim
        self.temporal_pooling = temporal_pooling
        self.upsample_mode = upsample_mode
        self.regression_mode = regression_mode
        self.freeze_pretrained = freeze_pretrained

        in_channels_actual = input_channels if input_channels is not None else in_channels
        self.input_adapter = (
            nn.Identity()
            if in_channels_actual == in_channels
            else nn.Conv1d(in_channels_actual, in_channels, kernel_size=1)
        )

        self.wavelet_decomp = SoftGateWaveletDecomp(
            in_channels=in_channels,
            max_level=max_level,
            kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_separate_channel,
            ffn_ratio=4.0,
            ffn_kernel_size=3,
            ffn_drop=0.1,
        )
        self.patch_embed = PatchEmbed(
            input_channels=1,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
        )
        self.pos_embed = (
            PositionEmbedding(embed_dim=embed_dim, pos_type=pos_embed_type)
            if use_pos_embed
            else None
        )
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        hidden_dims = [head_hidden_dim] if head_hidden_dim else None
        if self.regression_mode == "temporal":
            self.head = TemporalRegressionHead(
                embed_dim=embed_dim,
                output_dim=out_channels,
                hidden_dims=hidden_dims,
                dropout=head_dropout,
            )
        elif self.regression_mode == "pooled":
            self.head = RegressionHead(
                embed_dim=embed_dim,
                output_dim=out_channels,
                hidden_dims=hidden_dims,
                dropout=head_dropout,
                pooling="mean",
            )
        else:
            raise ValueError(f"Unknown regression_mode: {self.regression_mode}")

        if pretrained_path is not None:
            load_pretrained_feature_extractor(
                self, pretrained_path=pretrained_path, strict=pretrained_strict
            )
        if self.freeze_pretrained:
            for name, param in self.named_parameters():
                if name.startswith(("input_adapter", "head")):
                    continue
                param.requires_grad = False

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        wave_spec = self.wavelet_decomp(x)
        wave_2d = wave_spec.unsqueeze(1)
        tokens = self.patch_embed(wave_2d)
        if self.pos_embed is not None:
            p_f, p_t = self.patch_size
            freq_bands = (self.max_level + 1) * self.in_channels
            freq_size = freq_bands // p_f
            time_size = wave_spec.shape[-1] // p_t
            tokens = self.pos_embed(tokens, freq_size=freq_size, time_size=time_size)
        return tokens

    def _tokens_to_time(self, tokens: torch.Tensor, time_steps: int) -> torch.Tensor:
        bsz, num_tokens, dim = tokens.shape
        p_f, p_t = self.patch_size
        freq_bands = (self.max_level + 1) * self.in_channels
        freq_size = freq_bands // p_f
        time_size = time_steps // p_t
        if freq_size * time_size != num_tokens:
            raise ValueError(
                "Token count mismatch: "
                f"{num_tokens} vs expected {freq_size * time_size}."
            )
        tokens = tokens.view(bsz, freq_size, time_size, dim)
        if self.temporal_pooling == "mean":
            tokens = tokens.mean(dim=1)
        elif self.temporal_pooling == "max":
            tokens = tokens.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown temporal_pooling: {self.temporal_pooling}")
        features = tokens.permute(0, 2, 1)
        if time_size != time_steps:
            align = False if self.upsample_mode in {"linear", "bilinear"} else None
            features = F.interpolate(
                features,
                size=time_steps,
                mode=self.upsample_mode,
                align_corners=align,
            )
        return features.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input EMG
        Returns:
            [B, out_channels, T] regression output
        """
        x = self.input_adapter(x)
        tokens = self._prepare_tokens(x)
        encoded = self.encoder(tokens)
        if self.regression_mode == "temporal":
            time_features = self._tokens_to_time(encoded, x.shape[-1])
            preds = self.head(time_features)
            return preds.permute(0, 2, 1)
        pooled = self.head(encoded)
        return pooled
