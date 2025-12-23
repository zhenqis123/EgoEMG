from __future__ import annotations

import torch

from emg2pose.kinematics import load_default_hand_model
from emg2pose.UmeTrack.lib.common.pytorch3d_transforms_so3 import so3_exp_map


def get_joint_rotation_axes(num_joints: int = 20) -> torch.Tensor:
    hand_model = load_default_hand_model()
    axes = hand_model.joint_rotation_axes[:num_joints].clone()
    norms = torch.linalg.norm(axes, dim=-1, keepdim=True).clamp(min=1e-8)
    return axes / norms


def angles_to_axis_angle(joint_angles: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    return joint_angles[..., None] * axes


def axis_angle_to_angles(axis_angle: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    return (axis_angle * axes).sum(dim=-1)


def axis_angle_to_rot6d(axis_angle: torch.Tensor) -> torch.Tensor:
    flat = axis_angle.reshape(-1, 3)
    rot = so3_exp_map(flat)
    rot6d = rot[:, :, :2].reshape(-1, 6)
    return rot6d.reshape(*axis_angle.shape[:-1], 6)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_matrix_to_axis_angle(rot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos)
    skew = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )
    denom = 2.0 * torch.sin(angle).unsqueeze(-1)
    axis = skew / denom.clamp(min=eps)
    axis_angle = axis * angle.unsqueeze(-1)
    small = angle < eps
    axis_angle = torch.where(small.unsqueeze(-1), 0.5 * skew, axis_angle)
    return axis_angle


def rot6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    rot = rot6d_to_matrix(rot6d)
    return rotation_matrix_to_axis_angle(rot)
