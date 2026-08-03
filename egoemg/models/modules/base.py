# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn.functional import interpolate


class BaseModule(nn.Module):
    """
    Pose module consisting of a network with a left and right context. Predictions span
    the inputs[left_context : -right_context], and are upsampled to match the sample
    rate of the inputs.
    """

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module | None = None,
        out_channels: int = 20,
        provide_initial_pos: bool = False,
    ):
        super().__init__()
        # Backward compat: keep `network` attribute while exposing `featurizer`.
        self.featurizer = featurizer
        self.decoder = decoder
        self.out_channels = out_channels
        self.provide_initial_pos = provide_initial_pos

        self.left_context = featurizer.left_context
        self.right_context = featurizer.right_context

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        emg = batch["emg"]
        joint_angles = batch["joint_angles"]
        mask = batch["label_valid_mask"]

        # Get initial position
        initial_pos = joint_angles[..., self.left_context]
        if not self.provide_initial_pos:
            initial_pos = torch.zeros_like(initial_pos)

        # Generate prediction
        pred = self._predict_pose(emg, initial_pos)

        # Slice joint angles to match the span of the predictions
        start = self.left_context
        stop = None if self.right_context == 0 else -self.right_context
        joint_angles = joint_angles[..., slice(start, stop)]
        mask = mask[..., slice(start, stop)]

        # Match the sample rate of the predictions to that of the joint angles
        n_time = joint_angles.shape[-1]
        pred = self.align_predictions(pred, n_time)
        mask = self.align_mask(mask, n_time)
        return pred, joint_angles, mask

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        if self.decoder is None:
            raise NotImplementedError(
                "BaseModule requires a decoder or an override of _predict_pose()."
            )
        features = self.featurizer(emg)  # BCT
        return self.decoder(features)

    def align_predictions(self, pred: torch.Tensor, n_time: int):
        """Temporally resamples predictions to match the length of targets."""
        return interpolate(pred, size=n_time, mode="linear")

    def align_mask(self, mask: torch.Tensor, n_time: int):
        """Temporally resample mask to match the length of targets."""
        mask = mask[:, None].to(torch.float32)
        aligned = interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)
