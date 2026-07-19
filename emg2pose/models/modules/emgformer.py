# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import torch
from torch import nn
from torch.nn import functional as F

from emg2pose.models.modules.base import BaseModule
from emg2pose.models.modules._pooling import TemporalAttentionPool


def _debug_steps_enabled() -> bool:
    return os.environ.get("EMG2POSE_DEBUG_STEPS", "0").lower() in {"1", "true", "yes"}


class Emg2PoseFormer(BaseModule):
    """Transformer-based module agnostic to task type; head decides the semantics.

    When ``center_supervised=True``, temporal attention pooling aggregates the
    decoder output into a single center-frame vector before the head, returning
    a (preds, targets, mask) tuple compatible with the lightning module.
    """

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        head: nn.Module,
        out_channels: int = 20,
        provide_initial_pos: bool = False,
        center_supervised: bool = False,
        supervise_at_prediction_rate: bool = False,
    ):
        super().__init__(
            featurizer=featurizer,
            decoder=decoder,
            out_channels=out_channels,
            provide_initial_pos=provide_initial_pos,
        )
        self.head = head
        self.center_supervised = center_supervised
        self.supervise_at_prediction_rate = supervise_at_prediction_rate

        if center_supervised:
            feat_dim = None
            if hasattr(self.decoder, "output_proj") and isinstance(
                self.decoder.output_proj, nn.Linear
            ):
                feat_dim = int(self.decoder.output_proj.out_features)
            if feat_dim is None and hasattr(self.decoder, "input_proj"):
                ip = self.decoder.input_proj
                if isinstance(ip, nn.Linear):
                    feat_dim = int(ip.out_features)
                elif hasattr(ip, "in_features"):
                    feat_dim = int(ip.in_features)
            if feat_dim is None:
                feat_dim = getattr(self.featurizer, "out_channels", None)
            if feat_dim is None:
                raise ValueError(
                    "Cannot determine feature dimension for temporal_attn. "
                    "Ensure decoder.output_proj or featurizer.out_channels is set."
                )
            self.temporal_attn = TemporalAttentionPool(feat_dim)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        debug = _debug_steps_enabled()
        rank = int(os.environ.get("RANK", "0"))
        t0 = time.perf_counter()
        last_t = t0

        def mark(name: str, tensor: torch.Tensor | None = None) -> None:
            nonlocal last_t
            if not debug:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            shape = f" shape={tuple(tensor.shape)}" if tensor is not None else ""
            print(
                f"[emg2pose-debug][rank={rank}] model.forward {name}{shape}: "
                f"+{now - last_t:.3f}s total={now - t0:.3f}s",
                flush=True,
            )
            last_t = now

        emg = batch["emg"]
        mark("input", emg)
        features = self.featurizer(emg)  # BCT_feat
        mark("featurizer", features)
        decoded = self.decoder(features)
        mark("decoder", decoded)

        if self.center_supervised:
            # Temporal attention pooling: learn which time steps inform center frame
            emg_pooled = self.temporal_attn(decoded)  # (B, C)
            preds = self.head(emg_pooled.unsqueeze(-1))  # (B, 22, 1)
            mark("head", preds)

            if "joint_angles" in batch and "label_valid_mask" in batch:
                targets = batch["joint_angles"]
                mask = batch["label_valid_mask"]
                if mask.ndim >= 2 and mask.shape[-1] > 1:
                    mask = mask.float().mean(dim=-1, keepdim=True)
                elif mask.ndim >= 2:
                    mask = mask[..., :1]
                return preds, targets, mask
            return preds

        preds = self.head(decoded)
        mark("head", preds)
        if self.supervise_at_prediction_rate and "joint_angles" in batch:
            joint_angles = batch["joint_angles"]
            mask = batch["label_valid_mask"]
            start = self.left_context
            stop = None if self.right_context == 0 else -self.right_context
            targets = joint_angles[..., slice(start, stop)]
            mask = mask[..., slice(start, stop)]
            targets = F.interpolate(
                targets.float(),
                size=preds.shape[-1],
                mode="linear",
                align_corners=False,
            )
            if mask.ndim == 2:
                mask = F.interpolate(
                    mask[:, None].float(),
                    size=preds.shape[-1],
                    mode="nearest",
                ).squeeze(1).to(torch.bool)
            elif mask.ndim == 3:
                mask = F.interpolate(
                    mask.float(),
                    size=preds.shape[-1],
                    mode="nearest",
                ).to(torch.bool)
            mark("target_downsample", targets)
            return preds, targets, mask
        return preds
