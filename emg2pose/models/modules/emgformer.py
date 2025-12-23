# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from emg2pose.models.modules.base import BaseModule


class Emg2PoseFormer(BaseModule):
    """Transformer-based module agnostic to task type; head decides the semantics."""

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
    ) -> torch.Tensor:
        emg = batch["emg"]
        features = self.featurizer(emg)  # BCT_feat
        decoded = self.decoder(features)
        preds = self.head(decoded)
        return preds
