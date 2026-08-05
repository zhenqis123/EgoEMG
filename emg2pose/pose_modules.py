# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from emg2pose.constants import EMG_SAMPLE_RATE
from emg2pose.networks import SequentialLSTM

from torch import nn
from torch.nn.functional import interpolate


class BasePoseModule(nn.Module):
    """
    Pose module consisting of a network with a left and right context. Predictions span
    the inputs[left_context : -right_context], and are upsampled to match the sample
    rate of the inputs.
    """

    def __init__(
        self,
        network: nn.Module,
        out_channels: int = 20,
    ):
        super().__init__()
        self.network = network
        self.out_channels = out_channels

        self.left_context = network.left_context
        self.right_context = network.right_context

    def forward(
        self, batch: dict[str, torch.Tensor], provide_initial_pos: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        emg = batch["emg"]
        joint_angles = batch["joint_angles"]
        no_ik_failure = batch["no_ik_failure"]

        # Get initial position
        initial_pos = joint_angles[..., self.left_context]
        if not provide_initial_pos:
            initial_pos = torch.zeros_like(initial_pos)

        # Generate prediction
        pred = self._predict_pose(emg, initial_pos)

        # Slice joint angles to match the span of the predictions
        start = self.left_context
        stop = None if self.right_context == 0 else -self.right_context
        joint_angles = joint_angles[..., slice(start, stop)]
        no_ik_failure = no_ik_failure[..., slice(start, stop)]

        # Match the sample rate of the predictions to that of the joint angles
        n_time = joint_angles.shape[-1]
        pred = self.align_predictions(pred, n_time)
        no_ik_failure = self.align_mask(no_ik_failure, n_time)

        return pred, joint_angles, no_ik_failure

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        raise NotImplementedError

    def align_predictions(self, pred: torch.Tensor, n_time: int):
        """Temporally resamples predictions to match the length of targets."""
        return interpolate(pred, size=n_time, mode="linear")

    def align_mask(self, mask: torch.Tensor, n_time: int):
        """Temporally resample mask to match the length of targets."""
        # 2D Inputs don't work for interpolate(), so we add a dummy channel dimension
        mask = mask[:, None].to(torch.float32)
        aligned = interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)


class PoseModule(BasePoseModule):
    """
    Tracks pose by predicting posititions or velocities,
    optionally given the initial state.
    """

    def __init__(self, network: nn.Module, predict_vel: bool = False):
        super().__init__(network)
        self.predict_vel = predict_vel

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        pred = self.network(emg)  # BCT
        if self.predict_vel:
            pred = initial_pos[..., None] + torch.cumsum(pred, -1)
        return pred


class StatePoseModule(BasePoseModule):
    """
    Tracks pose by predicting posititions or velocities, optionally given the initial
    state and conditioned on the previous state at each time point.
    """

    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        state_condition: bool = True,
        predict_vel: bool = False,
        rollout_freq: int = 50,
    ):
        super().__init__(network)
        self.decoder = decoder
        self.state_condition = state_condition
        self.predict_vel = predict_vel
        self.rollout_freq = rollout_freq

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):

        features = self.network(emg)  # BCT
        preds = [initial_pos]

        # Resample features to rollout frequency
        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)

        # Reset LSTM hidden state
        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()

        for t in range(features.shape[-1]):

            # Prepare decoder inputs
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)

            # Predict pose
            pred = self.decoder(inputs)
            if self.predict_vel:
                pred = pred + preds[-1]
            preds.append(pred)

        # Remove first pred, because it is the initial_pos (not a network prediction)
        return torch.stack(preds[1:], dim=-1)


class VEMG2PoseWithInitialState(BasePoseModule):
    """
    Predict pose for num_position_steps steps, then integrate the velocity thereafter.
    """

    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        num_position_steps: int,
        state_condition: bool = True,
        rollout_freq: int = 50,
    ):
        super().__init__(network)
        self.decoder = decoder
        self.num_position_steps = num_position_steps
        self.state_condition = state_condition
        self.rollout_freq = rollout_freq

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        features = self.network(emg)  # BCT

        # Resample features to rollout frequency
        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)

        # Reset LSTM hidden state
        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()

        # Compute num_position_steps at the new sample rate
        num_position_steps = round(
            self.num_position_steps * (self.rollout_freq / EMG_SAMPLE_RATE)
        )
        preds = [initial_pos]

        for t in range(features.shape[-1]):

            # Prepare decoder inputs
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)

            # Predict pose and velocity
            output = self.decoder(inputs)  # BC
            pos, vel = torch.split(output, output.shape[1] // 2, dim=1)

            # Predict pose for the first num_position_steps
            # then integrate velocity thereafter
            pred = pos if t < num_position_steps else preds[-1] + vel
            preds.append(pred)

        # Remove first pred, because it is the initial_pos (not a network prediction)
        return torch.stack(preds[1:], dim=-1)
