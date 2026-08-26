"""EgoEMG visualization: MANO mesh overlay on head-view frames.

Correct projection pipeline:
1. Decode raw MANO vertices from pose/beta with MANO-right semantics.
2. For left hand, mirror the raw MANO geometry into left-hand chirality.
3. Apply precomputed `world_R` / `world_t` to obtain world-frame vertices.
4. Read `T_W_Camera` from `mocap_head_transform` and invert to `T_C_W`.
5. Project with calibration intrinsics, then map calib pixels back to video pixels.

Important:
- EgoEMG stores both hands in MANO-right canonical parameterization.
- Left-hand `world_R/world_t` are defined on x-mirrored raw MANO geometry.
- `world_R/world_t` were fit on raw MANO vertices, not wrist-centered vertices.
- Do not use wrist pose/orientation to place EgoEMG MANO mesh in world space.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch

from egoemg.visualization.mesh_renderer import ManoMeshRenderer

CALIB_PATH = "reprojection_assets/GX010023_standard_calibration.json"
CALIB_SIZE = (3840, 3360)
VIDEO_SIZE = (1280, 720)
CROP_XYWH = (227, 0, 823, 720)
MIRROR_X_3 = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


def _calib_to_video(pts_calib: np.ndarray) -> np.ndarray:
    """Map projected points from calib resolution (3840x3360) to video pixels."""
    cx, cy, cw, ch = CROP_XYWH
    out = np.empty_like(pts_calib)
    out[..., 0] = pts_calib[..., 0] * (cw / CALIB_SIZE[0]) + cx
    out[..., 1] = pts_calib[..., 1] * (ch / CALIB_SIZE[1]) + cy
    return out


class EgoEmgVisualizer:
    """Render MANO mesh overlays using `world_R/world_t` and `T_W_Camera`."""

    def __init__(
        self,
        data_root: str | Path,
        mano_model_path: str | Path | None = None,
        device: str = "cuda",
    ):
        self.data_root = Path(data_root)
        self.device = device

        # Resolve MANO model path: explicit arg > $EGOEMG_ROOT/data/mano_data/models.
        if mano_model_path is None:
            mano_model_path = (
                Path(os.environ.get("EGOEMG_ROOT", "."))
                / "data" / "mano_data" / "models"
            )

        with open(self.data_root / CALIB_PATH, encoding="utf-8") as f:
            calib = json.load(f)
        self.K_calib = np.array(calib["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(
            calib["distortion_coefficients"], dtype=np.float64
        ).reshape(-1, 1)

        self._mano_model_path = str(mano_model_path)
        self.mano_right = smplx.MANO(
            model_path=self._mano_model_path,
            is_rhand=True,
            flat_hand_mean=False,
            use_pca=False,
            num_pca_comps=45,
        )
        self.mano_right.to(self.device)
        self.faces_right = self.mano_right.faces.copy()

        self.renderer = ManoMeshRenderer(
            K=self.K_calib,
            dist_coeffs=self.dist_coeffs,
            calib_size=CALIB_SIZE,
            crop_xywh=CROP_XYWH,
            video_size=VIDEO_SIZE,
        )

    @staticmethod
    def _mirror_raw_mano_points_x(points: np.ndarray) -> np.ndarray:
        return points * MIRROR_X_3.astype(points.dtype, copy=False)

    @staticmethod
    def _flip_face_winding(faces: np.ndarray) -> np.ndarray:
        return faces[:, [0, 2, 1]]

    def get_T_C_W(self, T_W_Camera: np.ndarray) -> np.ndarray:
        """Get camera-from-world transform from a 4x4 world-from-camera transform."""
        T_W_Camera = np.asarray(T_W_Camera, dtype=np.float64)
        if T_W_Camera.shape != (4, 4):
            raise ValueError(
                f"T_W_Camera must have shape (4, 4), got {T_W_Camera.shape}"
            )
        return np.linalg.inv(T_W_Camera)

    def project_world_points(
        self,
        points_world: np.ndarray,
        T_W_Camera: np.ndarray,
    ) -> np.ndarray:
        """Project world points to video pixel coordinates."""
        T_C_W = self.get_T_C_W(T_W_Camera)
        rvec = cv2.Rodrigues(T_C_W[:3, :3])[0]
        tvec = T_C_W[:3, 3].reshape(3, 1)
        pts_calib, _ = cv2.projectPoints(
            np.asarray(points_world, dtype=np.float64),
            rvec,
            tvec,
            self.K_calib,
            self.dist_coeffs,
        )
        return _calib_to_video(pts_calib.reshape(-1, 2))

    def get_mano_verts_local(
        self,
        pose_aa: np.ndarray,
        beta: np.ndarray,
        hand: str = "right",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode local MANO geometry in EgoEMG's canonical semantics."""
        mano = self.mano_right

        global_orient = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        hand_pose = torch.tensor(
            pose_aa[3:48], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        betas = torch.tensor(beta, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = mano(global_orient=global_orient, hand_pose=hand_pose, betas=betas)

        verts = out.vertices[0].cpu().numpy()
        faces = self.faces_right.copy()
        if hand == "left":
            verts = self._mirror_raw_mano_points_x(verts)
            faces = self._flip_face_winding(faces)
        return verts, faces

    def get_mano_verts_world(
        self,
        pose_aa: np.ndarray,
        beta: np.ndarray,
        world_R: np.ndarray,
        world_t: np.ndarray,
        hand: str = "right",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode MANO and place it into world coordinates using precomputed transforms."""
        verts_local, faces = self.get_mano_verts_local(pose_aa, beta, hand)
        world_R = np.asarray(world_R, dtype=np.float64)
        world_t = np.asarray(world_t, dtype=np.float64)
        if world_R.shape != (3, 3):
            raise ValueError(f"world_R must have shape (3, 3), got {world_R.shape}")
        if world_t.shape != (3,):
            raise ValueError(f"world_t must have shape (3,), got {world_t.shape}")
        verts_world = verts_local @ world_R.T + world_t
        return verts_world, faces

    def render_frame(
        self,
        image: np.ndarray,
        pose_aa: np.ndarray,
        beta: np.ndarray,
        world_R: np.ndarray,
        world_t: np.ndarray,
        T_W_Camera: np.ndarray,
        hand: str = "right",
        color: tuple[float, float, float] = (0.4, 0.7, 1.0),
        alpha: float = 0.6,
    ) -> np.ndarray:
        """Render MANO mesh overlay on a single frame."""
        verts_world, faces = self.get_mano_verts_world(
            pose_aa, beta, world_R, world_t, hand
        )
        T_C_W = self.get_T_C_W(T_W_Camera)
        verts_cam = (T_C_W[:3, :3] @ verts_world.T).T + T_C_W[:3, 3]
        return self.renderer.render_overlay(image, verts_cam, faces, color=color, alpha=alpha)
