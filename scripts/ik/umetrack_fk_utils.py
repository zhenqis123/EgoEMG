"""UmeTrack FK utilities for Manus-to-angle fitting.

Provides FK computation, joint limits, and Manus→UmeTrack landmark mapping.
"""

import torch
import numpy as np
from pathlib import Path
from egoemg.kinematics import load_default_hand_model
from egoemg.UmeTrack.lib.common.hand_skinning import skin_landmarks
from egoemg.UmeTrack.lib.common.hand import HandModel

# ── UmeTrack landmark → Manus keypoint (25 nodes) mapping ──
# Manus 25 nodes: nodes 5,10,15,20 are Manus-specific extra nodes
# (not real joints, absent from SDK hand21). hand21 maps 1:1 to UmeTrack's
# first 20 landmarks (L0-L19), with the following semantics:
#   Four fingers: root→PIP(Manus), pip→IP(Manus), dip→DIP, tip→TIP
#   Thumb: root→PIP(Manus, skip extra node 1), dip→DIP, tip→TIP
# L20 wrist2 has no Manus counterpart and is excluded entirely.

MANUS_TO_UMETRACK_MAP: list[int] = [
    4,   # L0: thumb tip → Manus 4 (TIP)
    9,   # L1: index tip → Manus 9 (TIP)
    14,  # L2: middle tip → Manus 14 (TIP)
    19,  # L3: ring tip → Manus 19 (TIP)
    24,  # L4: pinky tip → Manus 24 (TIP)
    0,   # L5: wrist → Manus 0
    2,   # L6: thumb root → Manus 2 (PIP, skip extra node 1)
    3,   # L7: thumb dip → Manus 3 (DIP)
    6,   # L8: index root → Manus 6 (PIP)
    7,   # L9: index pip → Manus 7 (IP)
    8,   # L10: index dip → Manus 8 (DIP)
    11,  # L11: middle root → Manus 11 (PIP)
    12,  # L12: middle pip → Manus 12 (IP)
    13,  # L13: middle dip → Manus 13 (DIP)
    16,  # L14: ring root → Manus 16 (PIP)
    17,  # L15: ring pip → Manus 17 (IP)
    18,  # L16: ring dip → Manus 18 (DIP)
    21,  # L17: pinky root → Manus 21 (PIP)
    22,  # L18: pinky pip → Manus 22 (IP)
    23,  # L19: pinky dip → Manus 23 (DIP)
]

NUM_ACTIVE_LANDMARKS = 20  # L0-L19 only, L20 wrist2 excluded entirely

LM_WEIGHTS = torch.ones(NUM_ACTIVE_LANDMARKS, dtype=torch.float32)

# Manus right-hand → UmeTrack left-hand coordinate transform.
# Step 1: Mirror X (right→left): (x,y,z) → (-x,y,z)
# Step 2: Rotate frame (-90° around Y, Manus +Z→UmeTrack +X): (-x,y,z) → (z,y,x)
# Combined: (x,y,z) → (z,y,x)
MANUS_TO_UMETRACK_ROTATION = torch.tensor([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
], dtype=torch.float32)


def hand_model_to(hand_model: HandModel, device: torch.device) -> HandModel:
    """Move all tensor fields of a HandModel to the given device."""
    def _to(t):
        return t.to(device) if isinstance(t, torch.Tensor) else t
    return HandModel(
        joint_rotation_axes=_to(hand_model.joint_rotation_axes),
        joint_rest_positions=_to(hand_model.joint_rest_positions),
        joint_frame_index=_to(hand_model.joint_frame_index),
        joint_parent=_to(hand_model.joint_parent),
        joint_first_child=_to(hand_model.joint_first_child),
        joint_next_sibling=_to(hand_model.joint_next_sibling),
        landmark_rest_positions=_to(hand_model.landmark_rest_positions),
        landmark_rest_bone_weights=_to(hand_model.landmark_rest_bone_weights),
        landmark_rest_bone_indices=_to(hand_model.landmark_rest_bone_indices),
        hand_scale=_to(hand_model.hand_scale) if hand_model.hand_scale is not None else None,
        mesh_vertices=_to(hand_model.mesh_vertices) if hand_model.mesh_vertices is not None else None,
        mesh_triangles=hand_model.mesh_triangles,
        dense_bone_weights=_to(hand_model.dense_bone_weights) if hand_model.dense_bone_weights is not None else None,
        joint_limits=_to(hand_model.joint_limits) if hand_model.joint_limits is not None else None,
    )


def load_model() -> HandModel:
    return load_default_hand_model()


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (3,) → rotation matrix (3,3) via Rodrigues formula.

    Uses small-angle Taylor expansion when |aa| < eps to keep gradients flowing.
    """
    angle = torch.norm(aa)
    x, y, z = aa[0], aa[1], aa[2]
    K = torch.tensor([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ], dtype=aa.dtype, device=aa.device)
    I = torch.eye(3, dtype=aa.dtype, device=aa.device)

    eps = 1e-6
    if angle < eps:
        # R ≈ I + K + K²/2  (Rodrigues expansion: sin≈t, 1-cos≈t²/2)
        return I + K + 0.5 * (K @ K)

    axis = aa / angle
    K_norm = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=aa.dtype, device=aa.device)
    return I + torch.sin(angle) * K_norm + (1 - torch.cos(angle)) * (K_norm @ K_norm)


def make_wrist_transform(wrist_rotation_aa: torch.Tensor,
                         wrist_translation: torch.Tensor | None = None) -> torch.Tensor:
    """Build 4×4 wrist transform from 3D axis-angle rotation + optional 3D translation."""
    R = axis_angle_to_matrix(wrist_rotation_aa)
    T = torch.eye(4, dtype=wrist_rotation_aa.dtype, device=wrist_rotation_aa.device)
    T[:3, :3] = R
    if wrist_translation is not None:
        T[:3, 3] = wrist_translation
    return T


def fk_landmarks(angles_22d: torch.Tensor, hand_model: HandModel,
                 wrist_transform: torch.Tensor | None = None) -> torch.Tensor:
    """Forward kinematics: 22D angles → 21×3 landmark positions."""
    if wrist_transform is None:
        wrist_transform = torch.eye(4, dtype=angles_22d.dtype, device=angles_22d.device)
    return skin_landmarks(hand_model, angles_22d, wrist_transforms=wrist_transform)


def angles_20d_to_22d(angles_20d: torch.Tensor) -> torch.Tensor:
    """Pad 20D finger angles to 22D (add zero wrist angles at indices 20,21)."""
    if angles_20d.dim() == 0:
        angles_20d = angles_20d.unsqueeze(0)
    zeros = torch.zeros(2, dtype=angles_20d.dtype, device=angles_20d.device)
    return torch.cat([angles_20d, zeros], dim=-1)


def get_joint_limits(hand_model: HandModel) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (lower, upper) bounds for 20D finger angles (radians)."""
    limits = hand_model.joint_limits  # (22, 2)
    return limits[:20, 0].clone(), limits[:20, 1].clone()


def extract_manus_targets(manus_keypoints: np.ndarray) -> torch.Tensor:
    """Extract, reorder, and rotate Manus keypoints (25×3) → UmeTrack-aligned (20×3).

    manus_keypoints: (25, 3) numpy array in mm, Manus coordinate frame (fingers +Z).
    Returns (20, 3) tensor in UmeTrack coordinate frame (fingers +X).
    L20 wrist2 is excluded — it has no Manus counterpart.
    """
    targets = np.zeros((NUM_ACTIVE_LANDMARKS, 3), dtype=np.float32)
    for lm_idx, manus_idx in enumerate(MANUS_TO_UMETRACK_MAP):
        targets[lm_idx] = manus_keypoints[manus_idx]
    targets_rotated = targets @ MANUS_TO_UMETRACK_ROTATION.cpu().numpy().T
    return torch.from_numpy(targets_rotated)


def estimate_scale(manus_keypoints: np.ndarray, hand_model: HandModel) -> float:
    """Estimate hand scale from wrist-to-middle-fingertip distance.

    Uses the overall hand size rather than individual bone lengths, since
    UmeTrack and Manus have different internal bone proportions.
    """
    # Manus: wrist(0) → middle fingertip(14), in mm
    manus_dist = float(np.linalg.norm(manus_keypoints[14] - manus_keypoints[0]))

    # UmeTrack: wrist(L5) → middle fingertip(L2), in rest pose
    rest = hand_model.landmark_rest_positions.cpu().numpy()
    umetrack_dist = float(np.linalg.norm(rest[2] - rest[5]))

    if umetrack_dist < 1e-8:
        return 1.0
    return manus_dist / umetrack_dist
