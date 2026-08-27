"""MANO canonical camera-frame parameters (theta, tau).

The stored labels keep hand pose MANO-local (global orient zero) plus a
separate rigid world transform per hand and the head-camera pose. This
module folds those into the parameters a MANO layer consumes directly:

    verts_cam = MANO(global_orient=theta, hand_pose, betas, transl=tau)

for the right hand, exactly; for the left hand the output must be
x-mirrored (chirality cannot be absorbed by a rigid transform).

Derivation (verified numerically to 1e-4 mm against the legacy
decode->world->camera chain):

    theta = R_cw @ R_w                      (both hands share theta)
    tau_R  = R_cw @ t_w + t_cw + theta @ j0 - j0
    tau_L  = M @ (R_cw @ t_w + t_cw) + theta @ (M @ j0) - M @ j0

where [R_cw | t_cw] = inv(mocap_head_transform), [R_w | t_w] is the hand's
world transform, j0 is the posed model's root joint (the LBS pivot:
smplx rotates global_orient about j0, the legacy chain about the origin),
and M = diag(-1, 1, 1).
"""
from __future__ import annotations

import cv2
import numpy as np


MIRROR = np.array([-1.0, 1.0, 1.0])


def camera_from_head(head_t12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[R_cw | t_cw] = world->camera from a 12-float head transform."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(head_t12[:9], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(head_t12[9:12], dtype=np.float64)
    Tinv = np.linalg.inv(T)
    return Tinv[:3, :3], Tinv[:3, 3]


def mano_camera_params(head_t12: np.ndarray, world_t12: np.ndarray,
                       j0: np.ndarray, left: bool) -> dict:
    """Per-frame (theta as axis-angle, tau) for a direct MANO forward.

    j0: the root-joint position of MANO(pose, beta, global_orient=0)
    (i.e. decoder output joints[0]) — required because global_orient
    rotates about the root joint, not the coordinate origin.
    """
    R_cw, t_cw = camera_from_head(head_t12)
    R_w = np.asarray(world_t12[:9], dtype=np.float64).reshape(3, 3)
    t_w = np.asarray(world_t12[9:12], dtype=np.float64)
    theta = R_cw @ R_w
    b = R_cw @ t_w + t_cw
    j0 = np.asarray(j0, dtype=np.float64)
    if left:
        # The output mirror M conjugates the rotation: the mirrored mesh
        # must be rotated by M theta M, NOT theta (un-conjugated theta
        # flips the hand's orientation).
        Md = np.diag(MIRROR)
        theta = Md @ theta @ Md
        tau = b * MIRROR + theta @ j0 - j0
    else:
        tau = b + theta @ j0 - j0
    aa, _ = cv2.Rodrigues(theta)
    return {"theta_aa": aa.reshape(3), "tau": tau,
            "theta_rot": theta, "R_cw": R_cw, "t_cw": t_cw}


def batch_camera_params(head_t12: np.ndarray, world_t12: np.ndarray,
                        j0: np.ndarray, left: bool) -> dict:
    """Vectorised (N,12) arrays version of :func:`mano_camera_params`."""
    R_h = np.asarray(head_t12[:, :9], dtype=np.float64).reshape(-1, 3, 3)
    t_h = np.asarray(head_t12[:, 9:12], dtype=np.float64)
    R_cw = np.linalg.inv(R_h)
    t_cw = -np.einsum("nij,nj->ni", R_cw, t_h)
    R_w = np.asarray(world_t12[:, :9], dtype=np.float64).reshape(-1, 3, 3)
    t_w = np.asarray(world_t12[:, 9:12], dtype=np.float64)
    theta = np.einsum("nij,njk->nik", R_cw, R_w)
    b = np.einsum("nij,nj->ni", R_cw, t_w) + t_cw
    j0 = np.asarray(j0, dtype=np.float64)
    if left:
        Md = np.diag(MIRROR)
        theta = np.einsum("ik,nkj,jl->nij", Md, theta, Md)
        tau = b * MIRROR + np.einsum("nij,nj->ni", theta, j0) - j0
    else:
        tau = b + np.einsum("nij,nj->ni", theta, j0) - j0
    return {"theta_rot": theta, "tau": tau, "R_cw": R_cw, "t_cw": t_cw}
