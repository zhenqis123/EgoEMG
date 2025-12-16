# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from emg2pose.models.modules.base import BaseModule


class Emg2PoseFormer(BaseModule):
    """Transformer-based pose module outputting per-timestep predictions."""

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        head: nn.Module,
        out_channels: int = 20,
    ):
        super().__init__(featurizer=featurizer, decoder=decoder, out_channels=out_channels)
        self.head = head

    def forward(
        self, batch: dict[str, torch.Tensor], provide_initial_pos: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emg = batch["emg"]
        joint_angles = batch["joint_angles"]
        mask = batch["label_valid_mask"]

        start = self.left_context
        stop = None if self.right_context == 0 else -self.right_context
        targets = joint_angles[..., slice(start, stop)]
        mask = mask[..., slice(start, stop)]

        if targets.shape[-1] <= 0:
            raise RuntimeError(
                "Empty target span after applying left/right context. "
                f"left_context={self.left_context}, right_context={self.right_context}, "
                f"joint_angles_T={joint_angles.shape[-1]}"
            )

        features = self.featurizer(emg)  # BCT_feat
        decoded = self.decoder(features)
        preds = self.head(decoded)

        if preds.ndim == 2:
            preds = preds[..., None]
        n_time = targets.shape[-1]
        preds = self.align_predictions(preds, n_time)
        mask = self.align_mask(mask, n_time)

        return preds, targets, mask
