"""Coordinate normalization and rotations used by the marker teacher."""

import torch
import torch.nn.functional as F

def compute_local_frame_torch(landmarks: torch.Tensor, use_centroid: bool = False):
    """
    landmarks: [B,21,3]
    use_centroid: False -> O = 5-MCP average (wrist+index+middle+ring+pinky)
                    True  -> O = all 21 marker centroid
    Returns:
      O: [B,3]          local frame origin
      R: [B,3,3]        rotation matrix, rows are x_dir, y_dir, z_dir
    """
    # ---- 1. 用 wrist(0) 来定义旋转方向（保持原来的逻辑）----
    O_rot = landmarks[:, 0]                     # 只用于构造坐标轴，不再作为原点

    v1 = landmarks[:, 5] - O_rot                # [B,3]
    x_dir = F.normalize(v1, dim=1)              # [B,3]

    v2 = landmarks[:, 17] - O_rot               # [B,3]
    normal = torch.cross(v2, v1, dim=1)         # [B,3]
    z_dir = F.normalize(normal, dim=1)          # [B,3]

    y_dir = F.normalize(torch.cross(z_dir, x_dir, dim=1), dim=1)  # [B,3]

    # 把三个轴拼成 R（和原来完全一致）
    R = torch.stack([x_dir, y_dir, z_dir], dim=1)  # [B,3,3]

    # ---- 2. 原点切换 ----
    if use_centroid:
        O = landmarks.mean(dim=1)                 # [B,3] 全部 21 marker centroid
    else:
        O = (landmarks[:, 0] +
             landmarks[:, 5] +
             landmarks[:, 9] +
             landmarks[:, 13] +
             landmarks[:, 17]) / 5.0              # [B,3] 5-MCP 平均值

    return O, R


def transform_points_torch(verts: torch.Tensor, O: torch.Tensor, R: torch.Tensor):
    """
    verts: [B,N,3]
    O:     [B,3]
    R:     [B,3,3]
    返回:  transformed [B,N,3]
    """
    rel = verts - O.unsqueeze(1)                 # [B,N,3]
    # 一次 batched‐matmul 完成投影
    return torch.matmul(rel, R.transpose(1,2))


def transform_joints_coordinates_torch(keypoints: torch.Tensor, use_centroid: bool = False):
    """
    输入: keypoints [B,21,3] 或 [21,3] (torch.Tensor)
    输出: (transformed_kp, T_world2local)
    """
    original_ndim = keypoints.ndim
    if original_ndim == 2:
        keypoints_processed = keypoints[None, ...]
    else:
        keypoints_processed = keypoints

    O, R = compute_local_frame_torch(keypoints_processed, use_centroid=use_centroid)
    transformed_kp = transform_points_torch(keypoints_processed, O, R)  # [B,21,3]
    B = O.shape[0]
    t_world2local = -torch.matmul(R, O[..., None]).squeeze(-1)  # [B,3]
    T_world2local = torch.eye(4, device=keypoints.device, dtype=keypoints.dtype).repeat(B, 1, 1)
    T_world2local[:, :3, :3] = R
    T_world2local[:, :3, 3] = t_world2local
    if original_ndim == 2:
        transformed_kp = transformed_kp.squeeze(0)
        T_world2local = T_world2local.squeeze(0)
    return transformed_kp, T_world2local


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵（batch）转换为轴角表示（axis-angle）。

    参数:
        R (torch.Tensor): (..., 3, 3) 的旋转矩阵

    返回:
        axis_angle (torch.Tensor): (..., 3) 的轴角向量
    """
    eps = 1e-6

    # 计算旋转角度 theta
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)

    # 构造反对称矩阵分量
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    r = torch.stack([rx, ry, rz], dim=-1)  # (..., 3)

    # 处理 sin(theta)
    sin_theta = torch.sin(theta)
    sin_theta = torch.where(
        sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta
    )

    # 单位轴 * 角度
    axis = r / (2 * sin_theta).unsqueeze(-1)
    axis_angle = axis * theta.unsqueeze(-1)

    # 特殊处理 theta ≈ 0 的情况（小角度近似为零向量）
    axis_angle = torch.where(
        (theta < eps).unsqueeze(-1), torch.zeros_like(axis_angle), axis_angle
    )
    return axis_angle


def transform_hand_coordinates_torch(verts: torch.Tensor, landmarks: torch.Tensor,
                                     use_centroid: bool = False):
    """Transform mesh vertices using the same marker frame as inference."""
    origin, rotation = compute_local_frame_torch(landmarks, use_centroid)
    return transform_points_torch(verts, origin, rotation)
