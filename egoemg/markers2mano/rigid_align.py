"""Batched Kabsch alignment for optional MANO reconstruction checks."""

import torch

def compute_aligned_error_batched(
    pred_markers: torch.Tensor, gt_markers: torch.Tensor
):
    """Batched Kabsch alignment for tensors shaped ``(B, N, 3)``."""
    c_pred = pred_markers.mean(dim=1)
    c_gt = gt_markers.mean(dim=1)
    pred_centered = pred_markers - c_pred[:, None, :]
    gt_centered = gt_markers - c_gt[:, None, :]
    covariance = pred_centered.transpose(1, 2) @ gt_centered
    u, _, vt = torch.linalg.svd(covariance)
    rotation = vt.transpose(1, 2) @ u.transpose(1, 2)
    reflected = torch.linalg.det(rotation) < 0
    if reflected.any():
        vt = vt.clone()
        vt[reflected, -1, :] *= -1
        rotation = vt.transpose(1, 2) @ u.transpose(1, 2)
    translation = c_gt - torch.bmm(
        c_pred[:, None, :], rotation.transpose(1, 2)
    ).squeeze(1)
    aligned = torch.bmm(pred_markers, rotation.transpose(1, 2))
    aligned += translation[:, None, :]
    error_per_marker = torch.linalg.vector_norm(aligned - gt_markers, dim=-1)
    return error_per_marker, rotation, translation
