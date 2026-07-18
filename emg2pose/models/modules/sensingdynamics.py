# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from emg2pose.models.modules.base import BaseModule


class SensingDynamicsModule(BaseModule):
    """SensingDynamics pose estimation module [Sîmpetru et al., 2022a].

    Featurizer (2D conv over channels × time with SMU activations and circular
    channel padding) followed by a 3-layer MLP decoder.

    The module predicts *velocities* (not joint angles). Velocity prediction
    provides implicit temporal smoothing, removing the need for the original's
    150 ms moving average filter.
    """

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        out_channels: int = 20,
        provide_initial_pos: bool = False,
    ):
        super().__init__(
            featurizer=featurizer,
            decoder=decoder,
            out_channels=out_channels,
            provide_initial_pos=provide_initial_pos,
        )

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        # emg: (B, C_emg, T)
        features = self.featurizer(emg)  # (B, C_feat, T')
        # MLP decoder operates on the feature dimension (= last dim)
        features = features.swapaxes(-1, -2)  # (B, T', C_feat)
        vel = self.decoder(features)  # (B, T', out_channels)
        vel = vel.swapaxes(-1, -2)  # (B, out_channels, T')
        # Integrate velocity from initial position
        pred = initial_pos[..., None] + torch.cumsum(vel, dim=-1)
        return pred
