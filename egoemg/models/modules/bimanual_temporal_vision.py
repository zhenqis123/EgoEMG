"""Shared single-frame encoder plus a small center-supervised temporal block."""

from __future__ import annotations

from hydra.utils import instantiate

import torch
from torch import nn

from egoemg.checkpoints import load_branch


class TemporalFeatureFusion(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
        heads=4,
        dropout=0.1,
        feedforward_dim=None,
        num_frames=5,
        causal=False,
    ):
        super().__init__()
        if num_frames < 1 or num_frames % 2 != 1:
            raise ValueError("num_frames must be positive and odd")
        self.causal = bool(causal)
        self.middle = num_frames - 1 if self.causal else num_frames // 2
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, num_frames, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        self.attention = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            dim_feedforward=(feedforward_dim or hidden_dim * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(hidden_dim, feature_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, valid):
        tokens = self.input(features) + self.position
        if self.causal:
            n = features.shape[1]
            if not valid[:, -1].all():
                raise ValueError("Causal temporal vision requires a valid final frame")
            past = torch.ones(n, n, device=features.device, dtype=torch.bool).tril()
            allowed = past[None] & valid[:, None, :]
            # Invalid query rows need a finite self-attention path. Valid rows
            # still cannot attend any invalid key, preventing padding leakage.
            allowed |= (
                torch.eye(n, device=features.device, dtype=torch.bool)[None]
                & ~valid[:, :, None]
            )
            forbidden = (~allowed).repeat_interleave(
                self.attention.self_attn.num_heads, 0
            )
            tokens = self.attention(tokens, src_mask=forbidden)
        else:
            tokens = self.attention(tokens, src_key_padding_mask=~valid)
        return features[:, self.middle] + self.output(tokens[:, self.middle])


class BimanualTemporalVisionPose(nn.Module):
    bimanual_output = True
    optimizer_component_prefixes = {
        "vision": ("vision.",),
        "temporal": ("temporal.",),
    }

    def __init__(
        self,
        vision_checkpoint=None,
        hidden_dim=256,
        heads=4,
        dropout=0.1,
        feedforward_dim=None,
        num_frames=5,
        vision_config=None,
        causal=False,
    ):
        super().__init__()
        if (vision_checkpoint is None) == (vision_config is None):
            raise ValueError("Specify exactly one vision initialization source")
        if vision_checkpoint is not None:
            self.vision = load_branch(vision_checkpoint)
        else:
            self.vision = (
                vision_config
                if isinstance(vision_config, nn.Module)
                else instantiate(vision_config, _convert_="all")
            )
        self.num_frames = num_frames
        self.causal = bool(causal)
        if not getattr(self.vision, "bimanual_output", False):
            raise ValueError("Temporal vision requires a 44-angle model")
        feature_dim = self.vision.head[0].in_features
        self.temporal = TemporalFeatureFusion(
            feature_dim,
            hidden_dim,
            heads,
            dropout,
            feedforward_dim,
            num_frames,
            causal=self.causal,
        )
        self.vision.requires_grad_(True)
        self.left_context = self.right_context = 0

    @property
    def head(self):
        return self.vision.head

    def forward_features(self, batch):
        images = batch["vision_img"]
        if images.ndim != 5 or images.shape[1] != self.num_frames:
            raise ValueError("Unexpected RGB clip shape/frame count")
        b = images.shape[0]
        images = images.flatten(0, 1)
        if images.dtype == torch.uint8:
            images = (
                images.float() / 255 - self.vision.image_mean
            ) / self.vision.image_std
        features = self.vision.backbone(images).reshape(b, self.num_frames, -1)
        valid = batch["temporal_valid_mask"].bool()
        # The center is always an actual labeled sample, never padding.
        if valid.shape != (b, self.num_frames):
            raise ValueError("Unexpected temporal validity shape")
        return self.temporal(features, valid)

    def forward(self, batch):
        prediction = self.head(self.forward_features(batch)).reshape(-1, 22, 1)
        if "joint_angles" not in batch:
            return prediction
        return (
            prediction,
            batch["joint_angles"].reshape_as(prediction),
            batch["label_valid_mask"].reshape_as(prediction).bool(),
        )

    @staticmethod
    def align_mask(mask, n_time):
        return mask
