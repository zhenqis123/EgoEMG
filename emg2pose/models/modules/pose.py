# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn.functional import interpolate

from emg2pose.constants import EMG_SAMPLE_RATE
from emg2pose.models.modules.base import BaseModule
from emg2pose.models.decoders.lstm import SequentialLSTM


class PoseModule(BaseModule):
    """
    Tracks pose by predicting posititions or velocities,
    optionally given the initial state.
    """

    def __init__(self, featurizer: nn.Module, predict_vel: bool = False):
        super().__init__(featurizer, decoder=None)
        self.predict_vel = predict_vel

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        pred = self.featurizer(emg)  # BCT

        if self.predict_vel:
            pred = initial_pos[..., None] + torch.cumsum(pred, -1)
        return pred


class StatePoseModule(BaseModule):
    """
    Tracks pose by predicting posititions or velocities, optionally given the initial
    state and conditioned on the previous state at each time point.
    """

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        state_condition: bool = True,
        predict_vel: bool = False,
        rollout_freq: int = 50,
    ):
        super().__init__(featurizer, decoder)
        self.state_condition = state_condition
        self.predict_vel = predict_vel
        self.rollout_freq = rollout_freq

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):

        features = self.featurizer(emg)  # BCT
        preds = [initial_pos]

        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)

        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()

        for t in range(features.shape[-1]):

            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)

            pred = self.decoder(inputs)
            if self.predict_vel:
                pred = pred + preds[-1]
            preds.append(pred)

        return torch.stack(preds[1:], dim=-1)


class VEMG2PoseWithInitialState(BaseModule):
    """
    Predict pose for num_position_steps steps, then integrate the velocity thereafter.
    """

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        num_position_steps: int,
        state_condition: bool = True,
        rollout_freq: int = 50,
        head: nn.Module | None = None,
    ):
        super().__init__(featurizer, decoder)
        self.num_position_steps = num_position_steps
        self.state_condition = state_condition
        self.rollout_freq = rollout_freq
        self.head = head

    def forward(
        self, batch: dict[str, torch.Tensor], provide_initial_pos: bool
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emg = batch["emg"]
        joint_angles = batch['joint_angles']
        features = self.featurizer(emg)  # BCT
        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()
        # Get initial position
        initial_pos = joint_angles[..., self.left_context]
        if not provide_initial_pos:
            initial_pos = torch.zeros(joint_angles.shape[0], 256, device=joint_angles.device)
        preds = [initial_pos]
        for t in range(features.shape[-1]):

            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)
            output = self.decoder(inputs)  # BC
            pos, vel = torch.split(output, output.shape[1] // 2, dim=1)

            pred = pos if t < self.num_position_steps else preds[-1] + vel
            preds.append(pred)
        return self.head(torch.stack(preds[1:], dim=-1))

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        features = self.featurizer(emg)  # BCT

        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)

        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()

        num_position_steps = round(
            self.num_position_steps * (self.rollout_freq / EMG_SAMPLE_RATE)
        )
        preds = [initial_pos]

        for t in range(features.shape[-1]):

            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)

            output = self.decoder(inputs)  # BC
            pos, vel = torch.split(output, output.shape[1] // 2, dim=1)

            pred = pos if t < num_position_steps else preds[-1] + vel
            preds.append(pred)

        return torch.stack(preds[1:], dim=-1)
