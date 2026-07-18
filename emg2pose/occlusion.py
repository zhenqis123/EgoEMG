# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Self-occlusion quantification via z-buffer rasterization.

Projects a camera-space mesh (verts_cam, faces) onto the image plane,
rasterizes all triangles into a depth buffer, then checks per-vertex
visibility against the buffer.  The final score is area-weighted:
each triangle's 3-D area is distributed equally to its three vertices,
and the visible-surface ratio is the fraction of total area weight
accounted for by visible vertices.
"""

from __future__ import annotations

import numpy as np


def _project_vertices(
    verts_cam: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole-project camera-space vertices to pixel coordinates.

    Args:
        verts_cam: (V, 3) in camera frame: X right, Y down, Z forward.
        K: (3, 3) camera intrinsics.

    Returns:
        u: (V,) horizontal pixel coordinate (float).
        v: (V,) vertical pixel coordinate (float).
        valid: (V,) bool, True for vertices in front of camera.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X, Y, Z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]
    valid = Z > 1e-8
    u = np.full(len(verts_cam), -1.0, dtype=np.float64)
    v = np.full(len(verts_cam), -1.0, dtype=np.float64)
    inv_z = 1.0 / Z[valid]
    u[valid] = fx * X[valid] * inv_z + cx
    v[valid] = fy * Y[valid] * inv_z + cy
    return u, v, valid


def rasterize_z_buffer(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize mesh faces into a depth buffer.

    For each pixel, stores the smallest camera-space Z (closest point)
    among all faces that cover the pixel centre.

    Args:
        verts_cam: (V, 3) camera-space vertices (+Z forward).
        faces: (F, 3) triangle vertex indices.
        K: (3, 3) intrinsics.
        img_h, img_w: output buffer dimensions.

    Returns:
        zbuffer: (img_h, img_w) float64, inf where no face covers the pixel.
        u_proj: (V,) int, rounded pixel column of each vertex.
        v_proj: (V,) int, rounded pixel row of each vertex.
        valid_z: (V,) bool, vertices in front of camera.
        valid_img: (V,) bool, vertices whose projection falls inside the image.
    """
    u_float, v_float, valid_z = _project_vertices(verts_cam, K)
    V = len(verts_cam)
    u_proj = np.full(V, -1, dtype=np.int32)
    v_proj = np.full(V, -1, dtype=np.int32)
    u_proj[valid_z] = np.round(u_float[valid_z]).astype(np.int32)
    v_proj[valid_z] = np.round(v_float[valid_z]).astype(np.int32)
    valid_img = valid_z & (u_proj >= 0) & (u_proj < img_w) & (v_proj >= 0) & (v_proj < img_h)

    zbuffer = np.full((img_h, img_w), np.inf, dtype=np.float64)

    Z = verts_cam[:, 2]

    for fi, (i0, i1, i2) in enumerate(faces):
        if not (valid_img[i0] and valid_img[i1] and valid_img[i2]):
            continue

        u0, v0, z0 = float(u_float[i0]), float(v_float[i0]), float(Z[i0])
        u1, v1, z1 = float(u_float[i1]), float(v_float[i1]), float(Z[i1])
        u2, v2, z2 = float(u_float[i2]), float(v_float[i2]), float(Z[i2])

        # ── bounding box ──────────────────────────────────────────
        umin = max(0, int(np.floor(min(u0, u1, u2))))
        umax = min(img_w - 1, int(np.ceil(max(u0, u1, u2))))
        vmin = max(0, int(np.floor(min(v0, v1, v2))))
        vmax = min(img_h - 1, int(np.ceil(max(v0, v1, v2))))
        if umin > umax or vmin > vmax:
            continue

        # ── rasterize pixels inside triangle ──────────────────────
        for pv in range(vmin, vmax + 1):
            py = pv + 0.5  # pixel centre
            for pu in range(umin, umax + 1):
                px = pu + 0.5

                # Edge functions (positive inside, for CCW winding).
                e01 = (u1 - u0) * (py - v0) - (v1 - v0) * (px - u0)
                e12 = (u2 - u1) * (py - v1) - (v2 - v1) * (px - u1)
                e20 = (u0 - u2) * (py - v2) - (v0 - v2) * (px - u2)

                inside = (e01 >= 0.0 and e12 >= 0.0 and e20 >= 0.0) or \
                         (e01 <= 0.0 and e12 <= 0.0 and e20 <= 0.0)
                if not inside:
                    continue

                # Barycentric coordinates (perspective-correct would
                # interpolate 1/z, but for small hand triangles linear
                # interpolation of z is accurate enough).
                area = e01 + e12 + e20
                if abs(area) < 1e-18:
                    continue
                alpha = e12 / area
                beta = e20 / area
                gamma = e01 / area
                interp_z = alpha * z0 + beta * z1 + gamma * z2

                if interp_z < zbuffer[pv, pu]:
                    zbuffer[pv, pu] = interp_z

    return zbuffer, u_proj, v_proj, valid_z, valid_img


def check_vertex_visibility(
    u_px: int,
    v_px: int,
    z_vert: float,
    zbuffer: np.ndarray,
    depth_eps: float = 0.005,
    window_half: int = 2,
) -> bool:
    """Return True if a vertex is visible according to the z-buffer.

    A vertex is considered visible if *any* pixel in a
    (2*window_half+1) × (2*window_half+1) neighbourhood has a
    z-buffer depth within *depth_eps* of *z_vert*.
    """
    h, w = zbuffer.shape
    v0 = max(0, v_px - window_half)
    v1 = min(h, v_px + window_half + 1)
    u0 = max(0, u_px - window_half)
    u1 = min(w, u_px + window_half + 1)
    patch = zbuffer[v0:v1, u0:u1]
    if not np.isfinite(patch).any():
        return False
    return bool((np.abs(patch - z_vert) < depth_eps).any())


def _triangle_areas_3d(verts_cam: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute 3-D area of each triangle (half the cross-product norm)."""
    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def compute_vertex_area_weights(
    verts_cam: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Distribute 3-D triangle area equally to its three vertices.

    Returns (V,) float64 array of per-vertex area weights.
    A vertex that belongs to N faces accumulates N × (face_area / 3).
    """
    areas = _triangle_areas_3d(verts_cam, faces)
    weights = np.zeros(len(verts_cam), dtype=np.float64)
    for fi, (i0, i1, i2) in enumerate(faces):
        w = areas[fi] / 3.0
        weights[i0] += w
        weights[i1] += w
        weights[i2] += w
    return weights


def compute_self_occlusion(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    img_h: int,
    img_w: int,
    depth_eps: float = 0.005,
    window_half: int = 2,
) -> dict:
    """Compute area-weighted self-occlusion score for a single mesh.

    Args:
        verts_cam: (V, 3) camera-space vertices (+Z forward).
        faces: (F, 3) triangle indices.
        K: (3, 3) camera intrinsics.
        img_h, img_w: image dimensions in pixels.
        depth_eps: depth agreement threshold (metres).
        window_half: half-size of the local search window.

    Returns:
        Dict with keys:
          - occlusion_score:  float, 0 = fully visible, 1 = fully occluded.
          - visible_ratio:    float, fraction of area-weighted surface visible.
          - visible:          (V,) bool, per-vertex visibility.
          - area_weights:     (V,) float64, per-vertex area weight.
          - zbuffer:          (img_h, img_w) float64 depth buffer.
          - u_proj, v_proj:   (V,) int, rounded projection of each vertex.
    """
    zbuffer, u_proj, v_proj, valid_z, valid_img = rasterize_z_buffer(
        verts_cam, faces, K, img_h, img_w,
    )

    V = len(verts_cam)
    visible = np.zeros(V, dtype=bool)
    Z = verts_cam[:, 2]

    for i in range(V):
        if not valid_img[i]:
            continue
        visible[i] = check_vertex_visibility(
            int(u_proj[i]), int(v_proj[i]), float(Z[i]),
            zbuffer, depth_eps=depth_eps, window_half=window_half,
        )

    area_weights = compute_vertex_area_weights(verts_cam, faces)
    total = float(area_weights.sum())
    if total < 1e-12:
        visible_ratio = 0.0
    else:
        visible_ratio = float(area_weights[visible].sum()) / total

    return {
        "occlusion_score": 1.0 - visible_ratio,
        "visible_ratio": visible_ratio,
        "visible": visible,
        "area_weights": area_weights,
        "zbuffer": zbuffer,
        "u_proj": u_proj,
        "v_proj": v_proj,
    }
