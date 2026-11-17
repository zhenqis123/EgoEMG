"""Load the markers2mano network and convert complete marker frames."""

from pathlib import Path

import numpy as np
import torch

from .geometry import matrix_to_axis_angle, transform_joints_coordinates_torch
from .model import six_d_to_rot_matrix

MARKER_ORDER = (
    "HandM", "HandThumb", "ThumbM1", "ThumbM2", "ThumbM3",
    "HandIndex", "IndexM1", "IndexM2", "IndexM3",
    "HandMiddle", "MiddleM1", "MiddleM2", "MiddleM3",
    "HandRing", "RingM1", "RingM2", "RingM3",
    "HandPinky", "PinkyM1", "PinkyM2", "PinkyM3",
)

MODEL_CONFIG = dict(num_markers=21, beta_dim=10, hidden_dim=1280,
                  num_layers=12, heads=8, ffn_ratio=4)


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu"):
    """Strictly load the global-pose-head architecture, excluding MANO/optimizer state."""
    from .model import EfficientGraphTransformer4x

    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    hp = saved.get("hyper_parameters", {})
    for key, expected in MODEL_CONFIG.items():
        if hp.get(key, expected) != expected:
            raise ValueError(f"Expected markers2mano {key}={expected}, got {hp[key]}")
    if hp.get("use_centroid_origin", False):
        raise ValueError("conversion requires the five-marker palm origin")
    state = {key.removeprefix("model."): value
             for key, value in saved["state_dict"].items() if key.startswith("model.")}
    if not any(key.startswith("pose_head.") for key in state):
        raise ValueError("Expected the global pose_head checkpoint")
    model = EfficientGraphTransformer4x(
        num_markers=21, beta_dim=10, embed_dim=1280,
        num_layers=12, num_heads=8, dropout=0.0, ffn_ratio=4,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def convert_markers(
    model: torch.nn.Module,
    markers_m: np.ndarray,
    *,
    handedness: str,
    valid_mask: np.ndarray | None = None,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Convert (N,21,3) metre-valued surface markers using canonical MANO-right semantics.

    Incomplete/nonfinite/degenerate frames remain NaN with valid=False. No temporal
    interpolation is applied. world_to_canonical maps input markers into the
    network frame; global MANO mesh placement requires a separate rigid fit.
    """
    if handedness not in ("left", "right"):
        raise ValueError("handedness must be left or right")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    markers = np.asarray(markers_m, dtype=np.float32)
    if markers.ndim != 3 or markers.shape[1:] != (21, 3):
        raise ValueError("Expected markers_m with shape (N,21,3), in metres")
    count = len(markers)
    valid = np.isfinite(markers).all(axis=(1, 2))
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape == (count, 21):
            mask = mask.all(axis=1)
        elif mask.shape != (count,):
            raise ValueError("valid_mask must have shape (N,21) or (N,)")
        valid &= mask
    cross = np.cross(markers[:, 5] - markers[:, 0], markers[:, 17] - markers[:, 0])
    valid &= np.linalg.norm(cross, axis=1) > 1e-8
    pose = np.full((count, 48), np.nan, dtype=np.float32)
    beta = np.full((count, 10), np.nan, dtype=np.float32)
    transforms = np.full((count, 4, 4), np.nan, dtype=np.float32)
    indices = np.flatnonzero(valid)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            points = torch.from_numpy(markers[rows]).to(device)
            wrist = points[:, 0:1].clone()
            local, transform = transform_joints_coordinates_torch(points - wrist)
            transform[:, :3, 3] -= (transform[:, :3, :3] @ wrist.transpose(1, 2)).squeeze(-1)
            if handedness == "left":
                local[..., 2] *= -1
                transform[:, 2, :] *= -1
            pose_6d, betas = model(local)
            rotations = six_d_to_rot_matrix(pose_6d.reshape(-1, 16, 6))
            angles = matrix_to_axis_angle(rotations).reshape(-1, 48)
            angles[:, :3] = 0.0
            pose[rows] = angles.cpu().numpy()
            beta[rows] = betas.cpu().numpy()
            transforms[rows] = transform.cpu().numpy()
    valid &= np.isfinite(pose).all(axis=1) & np.isfinite(beta).all(axis=1)
    pose[~valid] = np.nan
    beta[~valid] = np.nan
    transforms[~valid] = np.nan
    mean_beta = beta[valid].mean(axis=0) if valid.any() else np.full(10, np.nan, np.float32)
    return dict(mano_pose=pose, mano_beta=beta, mano_beta_mean=mean_beta,
                valid=valid, world_to_canonical=transforms)
