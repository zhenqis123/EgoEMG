"""pyrender+OSMesa-based MANO mesh renderer with hardware-quality depth testing."""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import pyrender
import trimesh


def _calib_K_to_video_K(K_calib: np.ndarray,
                         crop_xywh: tuple[int, int, int, int],
                         calib_size: tuple[int, int]) -> np.ndarray:
    """Convert calibration intrinsics to video pixel intrinsics.

    The video is mapped from calib by:
      x_vid = x_calib * (crop_w / calib_w) + crop_x
      y_vid = y_calib * (crop_h / calib_h) + crop_y

    This maps a projection in calib pixels to video pixel coordinates.
    """
    cx, cy, cw, ch = crop_xywh
    cw_calib, ch_calib = calib_size

    scale_x = cw / cw_calib
    scale_y = ch / ch_calib

    K_video = K_calib.copy()
    K_video[0, 0] *= scale_x  # fx
    K_video[1, 1] *= scale_y  # fy
    K_video[0, 2] = K_calib[0, 2] * scale_x + cx  # cx
    K_video[1, 2] = K_calib[1, 2] * scale_y + cy  # cy
    return K_video


class ManoMeshRenderer:
    """Renders MANO mesh overlaid on an image using pyrender with OSMesa backend.

    Uses proper z-buffer depth testing (no Python face loop).
    Supports camera calibration intrinsics with distortion correction.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        K: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        calib_size: tuple[int, int] | None = None,
        crop_xywh: tuple[int, int, int, int] | None = None,
        video_size: tuple[int, int] | None = None,
    ):
        self.width = width
        self.height = height
        self.dist_coeffs = dist_coeffs

        if K is not None and calib_size and crop_xywh and video_size:
            self.K_video = _calib_K_to_video_K(K, crop_xywh, calib_size)
            self.fx = float(self.K_video[0, 0])
            self.fy = float(self.K_video[1, 1])
            self.cx = float(self.K_video[0, 2])
            self.cy = float(self.K_video[1, 2])
        else:
            self.fx = 2500
            self.fy = 2500
            self.cx = width / 2
            self.cy = height / 2

        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.camera = pyrender.IntrinsicsCamera(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, znear=0.01, zfar=10.0
        )
        self.scene = pyrender.Scene(
            ambient_light=np.array([1.0, 1.0, 1.0]),
            bg_color=np.array([0.0, 0.0, 0.0]),
        )
        self.scene.add(self.camera, pose=np.eye(4))
        self.mesh_node = None
        self._cached_mesh = None

    def __del__(self):
        if hasattr(self, "renderer"):
            try:
                self.renderer.delete()
            except Exception:
                pass

    def _opencv_to_opengl(self, verts_cam: np.ndarray) -> np.ndarray:
        """Convert OpenCV camera convention (+Z forward, +Y down) to OpenGL (-Z forward, +Y up)."""
        verts_gl = verts_cam.copy().astype(np.float32)
        verts_gl[:, 1] *= -1  # flip Y
        verts_gl[:, 2] *= -1  # flip Z
        return verts_gl

    def render_overlay(
        self,
        image: np.ndarray,
        verts_cam: np.ndarray,
        faces: np.ndarray,
        color: tuple[float, float, float] = (0.4, 0.7, 1.0),
        alpha: float = 0.6,
    ) -> np.ndarray:
        """Render mesh overlay on image using pyrender z-buffer depth testing.

        Args:
            image: (H, W, 3) uint8 RGB
            verts_cam: (V, 3) vertices in camera frame (for depth testing)
            faces: (F, 3) int face indices
            color: RGB mesh color [0,1]
            alpha: blend alpha
        Returns:
            blended: (H, W, 3) uint8
        """
        H, W = image.shape[:2]

        # Convert OpenCV -> OpenGL convention
        verts_render = self._opencv_to_opengl(verts_cam)

        # Reuse trimesh: update vertices in-place
        if self._cached_mesh is None:
            self._cached_mesh = trimesh.Trimesh(
                vertices=verts_render.copy(), faces=faces, process=False
            )
            self._cached_mesh.visual.vertex_colors = np.array(
                [int(c * 255) for c in color] + [255]
            )
        else:
            self._cached_mesh.vertices[:] = verts_render

        # Update pyrender scene
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(pyrender.Mesh.from_trimesh(self._cached_mesh))

        # Render
        rgba, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)

        # Guard
        if rgba.shape[0] != H or rgba.shape[1] != W or rgba.shape[2] == 0:
            return image

        # Blend
        if rgba.shape[2] >= 4:
            mask = rgba[:, :, 3:4].astype(np.float32) / 255.0 * alpha
            blended = rgba[:, :, :3]
        else:
            visible = (depth > 0).astype(np.float32)
            mask = visible[:, :, None] * alpha
            blended = rgba

        result = image.astype(np.float32) * (1 - mask) + blended.astype(np.float32) * mask
        return result.clip(0, 255).astype(np.uint8)
