# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from emg2pose.constants import (
    EMG_SAMPLE_RATE,
    FINGERS,
    JOINTS,
    LANDMARKS,
    NO_MOVEMENT_LANDMARKS,
    NUM_JOINTS,
    PD_GROUPS,
)

from emg2pose.kinematics import (
    forward_kinematics,
    load_default_hand_model,
    TorchHandModel,
)


class Metric:
    """Compute a dictionary of metrics from predicted and target joint angles."""

    weight: float = 0

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class AnglularDerivatives(Metric):
    """Mean absolute value of angular velocity, acceleration, and jerk."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        vel = torch.diff(pred, dim=-1)
        acc = torch.diff(vel, dim=-1)
        jerk = torch.diff(acc, dim=-1)

        mask = mask.unsqueeze(1).expand(-1, NUM_JOINTS, -1)  # BT -> BCT
        mask_vel = self.adjust_mask(mask)
        mask_acc = self.adjust_mask(mask_vel)
        mask_jerk = self.adjust_mask(mask_acc)

        # vel, acc, and jerk are in (radians / sample), so we multiply by the emg sample
        # rate to get (radians / second). Also take absolute value.
        return {
            f"{stage}_vel": vel[mask_vel].abs().mean() * EMG_SAMPLE_RATE,
            f"{stage}_acc": acc[mask_acc].abs().mean() * EMG_SAMPLE_RATE,
            f"{stage}_jerk": jerk[mask_jerk].abs().mean() * EMG_SAMPLE_RATE,
        }

    def adjust_mask(self, mask: torch.tensor):
        # Adjust mask to eliminate boundaries between IK failures
        # Note that this operation reduces the length of time by 1
        return ~F.max_pool1d((~mask).float(), kernel_size=2, stride=1).to(bool)


class AngleMAE(Metric):
    """Angular mean absolute error."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        no_ik_failure: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        mask = no_ik_failure.unsqueeze(1).expand(-1, NUM_JOINTS, -1)
        return {f"{stage}_mae": torch.nn.L1Loss()(pred[mask], target[mask])}


class PerFingerAngleMAE(Metric):
    """Angular mean absolute error for each finger."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        return {
            f"{stage}_mae_{finger}": self.get_error_for_finger(
                pred, target, mask, finger
            )
            for finger in FINGERS
        }

    @staticmethod
    def get_error_for_finger(pred, target, mask, finger: str):
        idxs = [j.index for j in JOINTS if finger in j.groups]
        mask = mask.unsqueeze(1).expand(-1, len(idxs), -1)
        return torch.nn.L1Loss()(pred[:, idxs][mask], target[:, idxs][mask])


class PDAngleMAE(Metric):
    """Angular mean absolute error grouped by proximal-distal joints."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        return {
            f"{stage}_mae_{group}": self.get_error_for_group(pred, target, mask, group)
            for group in PD_GROUPS
        }

    @staticmethod
    def get_error_for_group(pred, target, mask, group: str):
        idxs = [j.index for j in JOINTS if group in j.groups]
        mask = mask.unsqueeze(1).expand(-1, len(idxs), -1)
        return torch.nn.L1Loss()(pred[:, idxs][mask], target[:, idxs][mask])


class LandmarkDistances(Metric):
    """Mean Euclidian error for fingertips positions."""

    def __init__(self, downsampling: int = 40):
        self.downsampling = downsampling
        self.hand_model = TorchHandModel(load_default_hand_model())

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        if self.hand_model.device != pred.device:
            self.hand_model.to(pred.device)

        # Convert angles to 3D positions
        # We downsample in time to avoid OOM in forward_kinematics
        sl = slice(None, None, self.downsampling)
        pred_pos = forward_kinematics(pred[:, :, sl], self.hand_model)
        target_pos = forward_kinematics(target[:, :, sl], self.hand_model)
        mask_sliced = mask[:, sl]

        # Landmark distances
        lm_idxs = [lm.index for lm in LANDMARKS if lm.name not in NO_MOVEMENT_LANDMARKS]
        landmark_distance = self.get_mean_distance(
            pred_pos, target_pos, mask_sliced, lm_idxs
        )

        # Fingertip distances
        joint_idxs = [lm.index for lm in LANDMARKS if "fingertip" in lm.groups]
        fingertip_distance = self.get_mean_distance(
            pred_pos, target_pos, mask_sliced, joint_idxs
        )

        return {
            f"{stage}_fingertip_distance": fingertip_distance,
            f"{stage}_landmark_distance": landmark_distance,
        }

    def get_mean_distance(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        idxs: list,
    ):
        """
        Mean distance for given landmark idxs. Inputs are (batch, time, landmark, xyz).
        """
        mask = mask[..., None].expand(-1, -1, len(idxs))
        return torch.linalg.norm(pred[:, :, idxs] - target[:, :, idxs], dim=-1)[
            mask
        ].mean()


def get_default_metrics() -> list[Metric]:
    return [
        AngleMAE(),
        AnglularDerivatives(),
        PerFingerAngleMAE(),
        PDAngleMAE(),
        LandmarkDistances(),
    ]
