"""Shared helpers for dataset visualization (unified visualize_dataset.py).

Module-level imports are limited to stdlib + numpy + cv2 so that light
modes that do not need rendering do not pull in torch/smplx/pyrender/plotly; every
heavy dependency is imported lazily inside the function that needs it.

Consolidates boilerplate that used to be duplicated across the
scripts/viz/* dataset-visualization tools: memmap/metadata access,
calibration loading, world-to-video projection (wrapping the canonical
implementations in ``egoemg.datasets.egoemg_vision_dataset``), OpenCV
drawing primitives, GLB export, MANO/FK mesh decoding, the memmap
dataset factory, and crop LMDB access.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

# ── A. Headless runtime environment ─────────────────────────────────────────

_runtime_cache_root = Path(tempfile.gettempdir()) / "emg2pose_viz_runtime"


def setup_headless_environment() -> None:
    """Idempotently prepare a headless environment (pyrender/matplotlib)."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(_runtime_cache_root / "mpl"))
    os.environ.setdefault(
        "XDG_CACHE_HOME", str(_runtime_cache_root / "xdg-cache"))
    _runtime_cache_root.mkdir(parents=True, exist_ok=True)


# ── B. memmap / metadata access ─────────────────────────────────────────────

def load_manifest(memmap_dir: str | Path) -> dict[str, Any]:
    with (Path(memmap_dir) / "manifest.json").open() as f:
        return json.load(f)


def load_memmap(
    memmap_dir: str | Path,
    manifest: dict[str, Any],
    name: str,
    *,
    section: str = "fields",
) -> np.memmap:
    info = manifest[section][name]
    return np.memmap(
        Path(memmap_dir) / info["filename"],
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def load_metadata(memmap_dir: str | Path) -> np.lib.npyio.NpzFile:
    return np.load(Path(memmap_dir) / "metadata.npz", allow_pickle=False)


def decode_bytes(values: np.ndarray, *, strip_quotes: bool = False) -> list[str]:
    out: list[str] = []
    for v in values:
        s = v.decode("utf-8", errors="replace").rstrip("\x00") \
            if isinstance(v, (bytes, np.bytes_)) else str(v)
        if strip_quotes:
            s = s.strip("b'").strip('"')
        out.append(s)
    return out


# ── C. Calibration ──────────────────────────────────────────────────────────

DEFAULT_CALIB_FILENAME = "reprojection_assets/GX010023_standard_calibration.json"


@dataclass(frozen=True)
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int


def load_calibration(path: str | Path) -> Calibration:
    with Path(path).open() as f:
        calib = json.load(f)
    return Calibration(
        K=np.asarray(calib["camera_matrix"], dtype=np.float64),
        dist=np.asarray(
            calib["distortion_coefficients"], dtype=np.float64
        ).reshape(-1, 1),
        width=int(calib["image_width"]),
        height=int(calib["image_height"]),
    )


def resolve_calibration_path(
    data_root: str | Path,
    explicit: str | Path | None = None,
) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        return path
    calib_name = Path(DEFAULT_CALIB_FILENAME).name
    candidates = [
        Path(data_root) / DEFAULT_CALIB_FILENAME,
        Path(data_root) / "EgoEMG" / DEFAULT_CALIB_FILENAME,
        # Legacy LeRobot dataset tree (training_dataset_lerobot_full_NEW) and
        # the released preview package both carry the same calibration file.
        Path(data_root) / "training_dataset_lerobot_full_NEW" / DEFAULT_CALIB_FILENAME,
        Path(data_root) / "dataset_egoemg_preview" / "meta" / calib_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Calibration file not found in {[str(c) for c in candidates]} "
        f"(pass --calibration-path)")


# ── D. Transform helpers ────────────────────────────────────────────────────

def t12_to_matrix(t12: np.ndarray) -> np.ndarray:
    """12-float world-to-camera transform -> 4x4 matrix."""
    t12 = np.asarray(t12, dtype=np.float64).reshape(-1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = t12[:9].reshape(3, 3)
    T[:3, 3] = t12[9:12]
    return T


def t12_world_rt(t12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """12-float world transform [R(3x3) | t(3)] -> (R, t).

    Same layout as ``t12_to_matrix``; applied as
    ``verts_world = verts_local @ R.T + t`` (see
    :func:`verts_world_from_local`).
    """
    t12 = np.asarray(t12, dtype=np.float64).reshape(-1)
    return t12[:9].reshape(3, 3), t12[9:12]


def verts_world_from_local(
    verts_local: np.ndarray,
    world_R: np.ndarray,
    world_t: np.ndarray,
) -> np.ndarray:
    return verts_local @ np.asarray(world_R, dtype=np.float64).T \
        + np.asarray(world_t, dtype=np.float64).reshape(1, 3)


# ── E. Projection pipeline (canonical wrappers) ─────────────────────────────

def project_world_points(
    points_world: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from egoemg.datasets.egoemg_vision_dataset import _project_world_points
    return _project_world_points(points_world, T_W_C, K, dist)


def map_processed_points_to_raw(
    points_proc: np.ndarray,
    intrinsics_info: dict[str, Any],
) -> np.ndarray:
    from egoemg.datasets.egoemg_vision_dataset import (
        _map_processed_points_to_raw,
    )
    return _map_processed_points_to_raw(points_proc, intrinsics_info)


def build_intrinsics_and_frame_mapper(
    K: np.ndarray,
    dist: np.ndarray,
    calib_w: int,
    calib_h: int,
    video_w: int,
    video_h: int,
    first_frame_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from egoemg.datasets.egoemg_vision_dataset import (
        _build_intrinsics_and_frame_mapper,
    )
    return _build_intrinsics_and_frame_mapper(
        K, dist, calib_w, calib_h, video_w, video_h, first_frame_bgr)


def project_and_map(
    points_world: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    intrinsics_info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """world -> calib -> raw video pixels; returns (raw_px, depth_valid)."""
    proj, depth_valid = project_world_points(points_world, T_W_C, K, dist)
    raw = map_processed_points_to_raw(proj, intrinsics_info)
    return raw, depth_valid


def project_pinhole_K(verts_cam: np.ndarray, K: np.ndarray):
    """Project camera-space vertices through a pinhole K (no pose leg)."""
    v = np.asarray(verts_cam, dtype=np.float64)
    z = v[:, 2]
    u = (K[0, 0] * v[:, 0] / np.where(np.abs(z) < 1e-9, 1e-9, z)
         + K[0, 2])
    w = (K[1, 1] * v[:, 1] / np.where(np.abs(z) < 1e-9, 1e-9, z)
         + K[1, 2])
    return np.stack([u, w], axis=1), z > 0


def project_pinhole(
    points_world: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World -> pixels via an ideal pinhole camera (no distortion).

    Matches pyrender rendering on a frame undistorted with the same
    intrinsics (``cv2.undistort(frame, K, dist, None, K)``), so overlay
    layers drawn with this projector align pixel-exactly with a
    pyrender mesh.  Returns (px (N, 2), valid (N,) with Z > 0).
    """
    pts = np.asarray(points_world, dtype=np.float64)
    T_C_W = np.linalg.inv(T_W_C)
    vc = (T_C_W[:3, :3] @ pts.T).T + T_C_W[:3, 3]
    valid = vc[:, 2] > 1e-8
    px = np.full((len(pts), 2), -1.0, dtype=np.float64)
    px[valid, 0] = K[0, 0] * vc[valid, 0] / vc[valid, 2] + K[0, 2]
    px[valid, 1] = K[1, 1] * vc[valid, 1] / vc[valid, 2] + K[1, 2]
    return px, valid


def _project_draw_keypoints(
    image_bgr: np.ndarray,
    points_world: np.ndarray,
    valid: np.ndarray,
    color_bgr: tuple[int, int, int],
    projector: Any,
    *,
    radius: int = 3,
    edges: Sequence[tuple[int, int]] | None = None,
    label: str | None = None,
) -> np.ndarray:
    """Shared body of project_draw_keypoints / _pinhole variant."""
    kp = np.asarray(points_world, dtype=np.float64)
    kp_valid = np.asarray(valid, dtype=bool) & np.isfinite(kp).all(axis=1)
    if not kp_valid.any():
        return image_bgr
    pts_px, depth_valid = projector(kp)
    video_h, video_w = image_bgr.shape[:2]
    in_image = (
        (pts_px[:, 0] >= 0) & (pts_px[:, 0] < video_w)
        & (pts_px[:, 1] >= 0) & (pts_px[:, 1] < video_h))
    good = kp_valid & depth_valid & in_image
    if not good.any():
        return image_bgr
    return draw_skeleton(
        image_bgr, pts_px, good, color_bgr,
        label=label, radius=radius,
        edges=SKELETON_EDGES if edges is None else edges)


def project_draw_keypoints(
    image_bgr: np.ndarray,
    points_world: np.ndarray,
    valid: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    intrinsics_info: dict[str, Any],
    color_bgr: tuple[int, int, int],
    *,
    radius: int = 3,
    edges: Sequence[tuple[int, int]] | None = None,
    label: str | None = None,
) -> np.ndarray:
    """Project world-space keypoints onto the raw (distorted) image.

    A point is drawn only when it is marked ``valid``, finite,
    depth-valid and inside the image.  ``edges=None`` uses
    SKELETON_EDGES; pass ``edges=[]`` to draw joints only.
    """
    return _project_draw_keypoints(
        image_bgr, points_world, valid, color_bgr,
        lambda kp: project_and_map(kp, T_W_C, K, dist, intrinsics_info),
        radius=radius, edges=edges, label=label)


def project_draw_keypoints_pinhole(
    image_bgr: np.ndarray,
    points_world: np.ndarray,
    valid: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    radius: int = 3,
    edges: Sequence[tuple[int, int]] | None = None,
    label: str | None = None,
) -> np.ndarray:
    """project_draw_keypoints for the undistorted basis (pinhole + K).

    Use together with pyrender rendering on a frame undistorted with
    the same K so all overlay layers align pixel-exactly.
    """
    return _project_draw_keypoints(
        image_bgr, points_world, valid, color_bgr,
        lambda kp: project_pinhole(kp, T_W_C, K),
        radius=radius, edges=edges, label=label)


def intrinsics_info_to_video_K(
    intrinsics_info: dict[str, Any],
    K_calib: np.ndarray,
) -> np.ndarray:
    """Intrinsics matrix mapped to raw video pixel coordinates.

    The canonical intrinsics_info carries ``crop_xywh_on_video`` =
    [x0, y0, active_w, video_h] and ``processed_size`` = [calib_w, calib_h];
    the video height is the 4th crop component.
    """
    crop_xywh = intrinsics_info["crop_xywh_on_video"]
    proc_w = float(intrinsics_info["processed_size"][0])
    proc_h = float(intrinsics_info["processed_size"][1])
    crop_x = float(crop_xywh[0])
    crop_w = float(crop_xywh[2])
    video_h = float(crop_xywh[3])
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K_calib[0, 0] * crop_w / proc_w
    K[1, 1] = K_calib[1, 1] * video_h / proc_h
    K[0, 2] = K_calib[0, 2] * crop_w / proc_w + crop_x
    K[1, 2] = K_calib[1, 2] * video_h / proc_h
    return K


# ── F. Video reading ────────────────────────────────────────────────────────

def resolve_allintra_video_path(
    raw_video_path: str,
    *,
    data_root: str | Path,
    allintra_root: str | Path,
    suffix: str = "_allintra",
) -> Path:
    from egoemg.video_io import resolve_allintra_video_path as _resolve
    return _resolve(
        raw_video_path=raw_video_path,
        data_root=data_root,
        allintra_root=allintra_root,
        suffix=suffix,
    )


def try_resolve_allintra_video_path(
    raw_video_path: str,
    *,
    data_root: str | Path,
    allintra_root: str | Path,
    suffix: str = "_allintra",
) -> Path | None:
    try:
        return resolve_allintra_video_path(
            raw_video_path, data_root=data_root,
            allintra_root=allintra_root, suffix=suffix)
    except FileNotFoundError:
        return None


def open_video_reader(video_path: str | Path) -> Any:
    from decord import VideoReader, cpu
    return VideoReader(str(video_path), ctx=cpu(0))


def read_frame_bgr(reader: Any, frame_idx: int) -> np.ndarray:
    return np.ascontiguousarray(reader[frame_idx].asnumpy()[:, :, ::-1])


def clamp_frame_idx(reader: Any, frame_idx: int) -> int:
    return max(0, min(int(frame_idx), len(reader) - 1))


def open_mp4_writer(
    path: str | Path,
    fps: float,
    size: tuple[int, int],
) -> Any:
    """Open an MP4 VideoWriter, falling back avc1 -> mp4v.

    Raises RuntimeError when neither codec is available so a silent
    zero-byte output can never masquerade as a saved video.
    """
    import cv2
    out_fps = max(1, int(round(fps)))
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), out_fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(
        f"VideoWriter cannot open {path} (tried avc1, mp4v); "
        "install an H.264 encoder for MP4 output")


_pyrender_platform_reported = False


def ensure_pyrender_egl_bindings() -> None:
    """Bind the EGL device-enumeration functions PyOpenGL omits.

    PyOpenGL 3.1.10 does not bind ``eglQueryDevicesEXT`` /
    ``eglGetPlatformDisplayEXT``, so pyrender falls back to
    ``EGL_DEFAULT_DISPLAY``, which hangs in headless sessions.  Resolve
    them from the same libEGL the loader uses and patch pyrender's
    platform module before the first renderer is created.
    """
    import ctypes
    from ctypes import c_char_p, c_int, c_void_p, POINTER, CFUNCTYPE

    global _pyrender_platform_reported
    from pyrender.platforms import egl as pegl
    if pegl._eglQueryDevicesEXT is not None:
        _pyrender_platform_reported = True
        return
    lib = ctypes.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.restype = c_void_p
    lib.eglGetProcAddress.argtypes = [c_char_p]

    def bind(name: str, restype: Any, *argtypes: Any) -> Any:
        addr = lib.eglGetProcAddress(name.encode())
        if not addr:
            return None
        return CFUNCTYPE(restype, *argtypes)(addr)

    pegl._eglGetPlatformDisplayEXT = bind(
        "eglGetPlatformDisplayEXT", c_void_p, c_int, c_void_p, POINTER(c_int))
    pegl._eglQueryDevicesEXT = bind(
        "eglQueryDevicesEXT", c_int, c_int, POINTER(pegl._EGLDeviceEXT),
        POINTER(c_int))
    pegl._eglQueryDeviceStringEXT = bind(
        "eglQueryDeviceStringEXT", c_char_p, c_void_p, c_int)
    if pegl._eglQueryDevicesEXT is None:
        raise RuntimeError(
            "eglQueryDevicesEXT unavailable in libEGL; EGL GPU rendering "
            "cannot work on this machine")
    if not _pyrender_platform_reported:
        _pyrender_platform_reported = True
        print("[pyrender] patched EGL device-enumeration bindings "
              "(PyOpenGL 3.1.10 omission); platform=egl")


def make_pyrender_renderer(width: int, height: int) -> Any:
    """Create a pyrender OffscreenRenderer: EGL (GPU) preferred.

    EGL needs /dev/dri access (user in the video/render groups) plus the
    device-enumeration bindings from :func:`ensure_pyrender_egl_bindings`;
    on any failure it falls back to osmesa (software) and reports which
    platform is active.  pyrender reads PYOPENGL_PLATFORM at renderer
    creation, so switching the env var and retrying needs no reload.
    """
    import os
    import pyrender
    if os.environ.get("PYOPENGL_PLATFORM", "egl") == "egl":
        try:
            ensure_pyrender_egl_bindings()
            renderer = pyrender.OffscreenRenderer(width, height)
            print(f"[pyrender] EGL (GPU) renderer {width}x{height}")
            return renderer
        except Exception as exc:
            print(f"[pyrender] EGL unavailable ({exc}); "
                  "falling back to osmesa (software)")
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    return pyrender.OffscreenRenderer(width, height)


# ── G. OpenCV drawing primitives ────────────────────────────────────────────

SKELETON_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
# Hand skeleton plus palm edges (markers-video bone layout).
HAND_BONES = SKELETON_EDGES + [(0, 17), (5, 9), (9, 13), (13, 17)]

HAND_COLORS_BGR = {
    "right": (0, 180, 255),  # Orange
    "left": (255, 180, 0),   # Blue
}
HAND_COLORS_RGB = {
    "right": (255, 180, 0),
    "left": (0, 180, 255),
}
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_SHADOW_BGR = (20, 20, 20)


def draw_points(
    image_bgr: np.ndarray,
    points_xyc: np.ndarray,
    color_bgr: tuple[int, int, int],
    radius: int,
) -> np.ndarray:
    out = image_bgr.copy()
    for point in points_xyc:
        if point.shape[0] < 3 or point[2] <= 0:
            continue
        if not np.isfinite(point[:2]).all():
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(out, (x, y), radius, color_bgr, -1, lineType=cv2.LINE_AA)
    return out


def draw_bbox(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    color_bgr: tuple[int, int, int],
    line_width: int,
) -> np.ndarray:
    out = image_bgr.copy()
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox_xyxy]
    cv2.rectangle(out, (x0, y0), (x1, y1), color_bgr, line_width,
                  lineType=cv2.LINE_AA)
    return out


def draw_text_block(
    image_bgr: np.ndarray,
    lines: list[str],
    *,
    origin_xy: tuple[int, int] = (20, 30),
    line_height: int = 28,
    font_scale: float = 0.7,
) -> np.ndarray:
    out = image_bgr.copy()
    x, y = origin_xy
    for line in lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, TEXT_SHADOW_BGR, 3, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, TEXT_COLOR_BGR, 1, cv2.LINE_AA)
        y += line_height
    return out


def draw_skeleton(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    valid: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    label: str | None = None,
    radius: int = 3,
    edges: Sequence[tuple[int, int]] = SKELETON_EDGES,
) -> np.ndarray:
    """Draw skeleton edges and joints on an image."""
    out = image_bgr.copy()
    valid_bool = np.asarray(valid, dtype=bool)
    for i0, i1 in edges:
        if i0 >= len(pts) or i1 >= len(pts):
            continue
        if valid_bool[i0] and valid_bool[i1]:
            p0 = tuple(np.round(pts[i0]).astype(np.int32))
            p1 = tuple(np.round(pts[i1]).astype(np.int32))
            cv2.line(out, p0, p1, color_bgr, 2, lineType=cv2.LINE_AA)
    for i, (p, v) in enumerate(zip(pts, valid_bool)):
        if not v:
            continue
        center = tuple(np.round(p).astype(np.int32))
        cv2.circle(out, center, radius, color_bgr, -1, lineType=cv2.LINE_AA)
    if label is not None and valid_bool.any():
        cy, cx = pts[valid_bool].mean(axis=0).astype(np.int32)
        cv2.putText(out, label, (int(cx) + 10, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)
    return out


def draw_wireframe(
    image_bgr: np.ndarray,
    points_xy: np.ndarray,
    valid: np.ndarray,
    faces: np.ndarray,
    color_bgr: tuple[int, int, int],
    line_width: int,
) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        p0 = tuple(np.round(points_xy[i0]).astype(np.int32))
        p1 = tuple(np.round(points_xy[i1]).astype(np.int32))
        p2 = tuple(np.round(points_xy[i2]).astype(np.int32))
        cv2.line(image_bgr, p0, p1, color_bgr, line_width, lineType=cv2.LINE_AA)
        cv2.line(image_bgr, p1, p2, color_bgr, line_width, lineType=cv2.LINE_AA)
        cv2.line(image_bgr, p2, p0, color_bgr, line_width, lineType=cv2.LINE_AA)
    return image_bgr


# ── H. GLB export ───────────────────────────────────────────────────────────

def _mesh_part(vertices: np.ndarray, faces: np.ndarray,
               color_rgba: tuple[int, int, int, int]) -> Any:
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = list(color_rgba)
    return mesh


def _marker_sphere(center: np.ndarray, radius: float,
                   color_rgba: tuple[int, int, int, int]) -> Any:
    import trimesh
    s = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    s.apply_translation(np.asarray(center, dtype=np.float64))
    s.visual.vertex_colors = list(color_rgba)
    return s


def save_mesh_glb(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    path: str | Path,
) -> None:
    """Save a world-space mesh as GLB (icosphere fallback for bad input)."""
    if vertices_world is None or not np.isfinite(vertices_world).all():
        import trimesh
        trimesh.creation.icosphere(radius=0.001).export(str(path))
        return
    _mesh_part(vertices_world, faces, tuple(color) + (255,)).export(str(path))


def save_glb_with_markers(
    out_path: str | Path,
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    mesh_color: tuple[int, int, int, int] = (80, 200, 120, 220),
    gt_markers: np.ndarray | None = None,
    gt_marker_color: tuple[int, int, int, int] = (255, 215, 0, 255),
    gt_marker_radius: float = 0.004,
    pred_markers: np.ndarray | None = None,
    pred_marker_color: tuple[int, int, int, int] = (65, 105, 225, 200),
    pred_marker_radius: float = 0.003,
) -> None:
    import trimesh
    parts: list[Any] = [_mesh_part(verts, faces, mesh_color)]
    if gt_markers is not None:
        for pt in gt_markers:
            parts.append(_marker_sphere(pt, gt_marker_radius, gt_marker_color))
    if pred_markers is not None:
        for pt in pred_markers:
            parts.append(
                _marker_sphere(pt, pred_marker_radius, pred_marker_color))
    scene = trimesh.Scene(parts)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))


def save_comparison_glb(
    out_path: str | Path,
    fk_verts: np.ndarray,
    fk_faces: np.ndarray,
    mano_verts: np.ndarray,
    mano_faces: np.ndarray,
    *,
    fk_color: tuple[int, int, int, int] = (80, 140, 240, 220),
    mano_color: tuple[int, int, int, int] = (80, 220, 140, 220),
    offset_x: float = 0.15,
) -> None:
    """FK (blue, left) + MANO (green, right) side-by-side GLB."""
    import trimesh
    fk = _mesh_part(fk_verts, fk_faces, fk_color)
    mano = _mesh_part(mano_verts, mano_faces, mano_color)
    fk.apply_translation([-offset_x, 0.0, 0.0])
    mano.apply_translation([+offset_x, 0.0, 0.0])
    scene = trimesh.Scene([fk, mano])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))


# ── I. MANO / FK decoding ───────────────────────────────────────────────────

# 21 mocap marker vertex indices on the MANO mesh
# (same source as scripts/mano/infer_mano_for_egoemg.py).
MARKER_VERT_INDICES = np.asarray(
    [191, 88, 253, 708, 729, 144, 87, 295, 319, 220, 365, 407, 445,
     183, 477, 518, 556, 83, 589, 635, 673],
    dtype=np.int64,
)
_MIRROR_X_3 = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


class ManoMeshDecoder:
    """smplx MANO decoder producing (verts, faces) + surface markers.

    Left hand: x-mirror vertices and flip face winding (canonical
    EgoEmgVisualizer semantics).  Vertices are in MANO-local space;
    use :meth:`verts_world` to apply a world transform.
    """

    def __init__(
        self,
        mano_model_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        import smplx
        import torch
        self._device = device
        path = self._resolve_model_path(mano_model_path)
        self._mano = smplx.MANO(
            model_path=str(path),
            is_rhand=True,
            flat_hand_mean=False,
            use_pca=False,
            num_pca_comps=45,
        ).to(device)
        self._faces = self._mano.faces.astype(np.int64)
        self._torch = torch

    @staticmethod
    def _resolve_model_path(explicit: str | Path | None) -> Path:
        if explicit is not None:
            return Path(explicit)
        root = os.environ.get("EGOEMG_ROOT", "")
        if root:
            cand = Path(root) / "data" / "mano_data" / "models"
            if cand.is_dir():
                return cand
        raise FileNotFoundError(
            "MANO model not found; pass --mano-model-path or place the MANO "
            "models under $EGOEMG_ROOT/data/mano_data/models")

    def decode(
        self,
        pose_aa: np.ndarray,
        beta: np.ndarray,
        hand: str = "right",
    ) -> tuple[np.ndarray, np.ndarray]:
        """(verts_local (778,3), faces (1538,3)); left hand mirrored."""
        torch = self._torch
        global_orient = torch.zeros(1, 3, dtype=torch.float32,
                                    device=self._device)
        hand_pose = torch.tensor(
            np.asarray(pose_aa)[3:48], dtype=torch.float32,
            device=self._device).unsqueeze(0)
        betas = torch.tensor(
            np.asarray(beta), dtype=torch.float32,
            device=self._device).unsqueeze(0)
        with torch.no_grad():
            out = self._mano(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=betas,
            )
        verts = out.vertices[0].cpu().numpy()
        faces = self._faces
        if hand == "left":
            verts = verts * _MIRROR_X_3.astype(verts.dtype, copy=False)
            faces = faces[:, [0, 2, 1]]
        return verts, faces

    def decode_with_camera(self, theta_aa, pose_aa, beta, tau, hand):
        """MANO forward with explicit camera placement.

        Returns camera-space vertices (and faces via decoder semantics):
        right hand direct; left hand output is x-mirrored (chirality).
        theta_aa: 3-axis-angle of the global orientation; tau: camera-frame
        translation (see mano_camera_params.py for the derivation).
        """
        import torch
        torch = self._torch
        go = torch.tensor(np.asarray(theta_aa, dtype=np.float32),
                          device=self._device).unsqueeze(0)
        hp = torch.tensor(np.asarray(pose_aa, dtype=np.float32)[3:48],
                          device=self._device).unsqueeze(0)
        bt = torch.tensor(np.asarray(beta, dtype=np.float32),
                          device=self._device).unsqueeze(0)
        tr = torch.tensor(np.asarray(tau, dtype=np.float32),
                          device=self._device).unsqueeze(0)
        with torch.no_grad():
            out = self._mano(global_orient=go, hand_pose=hp, betas=bt,
                             transl=tr)
        verts = out.vertices[0].cpu().numpy()
        faces = self._faces
        if hand == "left":
            verts = verts * _MIRROR_X_3.astype(verts.dtype, copy=False)
            faces = faces[:, [0, 2, 1]]
        return verts, faces

    def root_joint(self, pose_aa, beta):
        """Root joint of the orient=0 model — the LBS pivot for theta."""
        import torch
        torch = self._torch
        hp = torch.tensor(np.asarray(pose_aa, dtype=np.float32)[3:48],
                          device=self._device).unsqueeze(0)
        bt = torch.tensor(np.asarray(beta, dtype=np.float32),
                          device=self._device).unsqueeze(0)
        go = torch.zeros(1, 3, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            out = self._mano(global_orient=go, hand_pose=hp, betas=bt)
        return out.joints[0, 0].cpu().numpy()

    def marker_vertices(self, verts_local: np.ndarray) -> np.ndarray:
        return verts_local[MARKER_VERT_INDICES]

    def verts_world(
        self,
        verts_local: np.ndarray,
        world_R: np.ndarray,
        world_t: np.ndarray,
    ) -> np.ndarray:
        return verts_world_from_local(verts_local, world_R, world_t)


def skin_mesh_from_angles(
    joint_angles: np.ndarray,
    *,
    flip: bool = False,
    profile: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """UmeTrack LBS mesh skinning (canonical classic.skin_mesh_from_angles)."""
    from egoemg.visualization.classic import skin_mesh_from_angles as _skin
    return _skin(joint_angles, user_profile=profile, flip=flip)


def rescale_mesh_span(verts: np.ndarray, target_span: float) -> np.ndarray:
    """Scale a mesh so its median vertex span equals ``target_span``."""
    span = float(np.median(verts.max(axis=0) - verts.min(axis=0)))
    if span > 1e-8:
        verts = verts * (target_span / span)
    return verts


def fk_mesh_world(
    joint_angles: np.ndarray,
    R_world: np.ndarray,
    t_world: np.ndarray,
    *,
    mirror_x: bool,
    anchor_verts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Skin an FK mesh from joint angles and place it in world space.

    UmeTrack skins in millimetres with a wrist origin that differs from
    MANO's: convert to metres and anchor vertex 0 (wrist) to the MANO
    wrist when ``anchor_verts`` (MANO local verts, same handedness) is
    given.  Empirically the stored joint angles pair with the mirrored
    profile for the RIGHT hand and the plain profile for the LEFT hand
    (nearest-neighbour check against the MANO meshes, mesh-mode smoke);
    mirror_profile does not flip triangle winding, so mirror_x also
    flips the faces.  Returns None when the angles are not finite /
    all-zero (no FK supervision for this row).
    """
    ja = np.asarray(joint_angles, dtype=np.float32)
    if not np.isfinite(ja).all() or np.abs(ja).sum() <= 0:
        return None
    try:
        fk_v, fk_f = skin_mesh_from_angles(joint_angles=ja[:20], flip=mirror_x)
        fk_f = np.asarray(fk_f)  # mirrored profile returns a torch Tensor
        fk_v = fk_v.astype(np.float64) / 1000.0  # mm -> m
        if mirror_x:
            fk_f = fk_f[:, [0, 2, 1]].copy()
        if anchor_verts is not None:
            anchor = np.asarray(anchor_verts, dtype=np.float64)
            fk_v = fk_v + (anchor[0] - fk_v[0])
        return verts_world_from_local(fk_v, R_world, t_world), fk_f
    except Exception:
        return None


# ── J. Dataset factory ──────────────────────────────────────────────────────

def make_memmap_dataset(
    *,
    memmap_dir: str | Path = "data/EgoEMG_full_memmap",
    hand: str = "right",
    window_length: int = 1000,
    stride: int | None = None,
    modalities: Sequence[str] = ("emg", "joint_angles", "mocap_hands", "mano"),
    mano_npy_dir: str | Path | None = None,
    emg_field_preference: str = "filtered",
    emg_layout: str = "target_hand",
    norm_mode: str | None = None,
    **kwargs: Any,
) -> Any:
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    if hand not in {"left", "right"}:
        raise ValueError(f"hand must be left/right, got {hand}")
    return EgoEmgMemmapDataset(
        memmap_dir=Path(memmap_dir),
        window_length=window_length,
        stride=stride or window_length,
        modalities=list(modalities),
        target_hand=hand,
        emg_field_preference=emg_field_preference,
        emg_layout=emg_layout,
        norm_mode=norm_mode,
        dataset_name="egoemg",
        mano_npy_dir=mano_npy_dir,
        **kwargs,
    )


def find_window_indices(ds: Any, ep_idx: int) -> list[int]:
    start = int(ds._block_cumsum[ep_idx])
    end = int(ds._block_cumsum[ep_idx + 1])
    return list(range(start, end))


def find_window_at_offset(ds: Any, ep_idx: int, offset: int) -> int | None:
    block_mask = np.asarray(ds._block_episode_idx) == ep_idx
    for bi in np.where(block_mask)[0]:
        w_start = int(ds._block_cumsum[bi])
        w_end = int(ds._block_cumsum[bi + 1])
        block_start = int(ds._block_start[bi])
        for w in range(w_start, w_end):
            rel = w - w_start
            start = block_start + rel * ds.stride
            end = start + ds.window_length
            if start <= offset < end:
                return w
    return None


def window_location(ds: Any, sample_idx: int) -> tuple[int, str, int]:
    """(episode_index, episode_id, window-center absolute frame)."""
    bi = int(np.searchsorted(ds._block_cumsum, sample_idx, side="right") - 1)
    ep_idx = int(ds._block_episode_idx[bi])
    block_start = int(ds._block_start[bi])
    rel = int(sample_idx - ds._block_cumsum[bi])
    start = block_start + rel * ds.stride
    center = start + ds.window_length // 2
    return ep_idx, ds._episode_id[ep_idx], center


# ── K. Crops LMDB ───────────────────────────────────────────────────────────

def read_crop_from_lmdb(lmdb_path: str | Path, key: str) -> np.ndarray | None:
    import lmdb
    from PIL import Image
    import io
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        txn = env.begin()
        jpeg = txn.get(key.encode())
    finally:
        env.close()
    if jpeg is None:
        return None
    img = Image.open(io.BytesIO(jpeg))
    return np.asarray(img)  # RGB


def list_lmdb_keys(lmdb_path: str | Path) -> list[str]:
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        txn = env.begin()
        return [k.decode() for k, _ in txn.cursor()]
    finally:
        env.close()


def load_episode_crop(
    crops_dir: str | Path,
    episode_id: str,
    frame_idx: int,
    hand_code: str,
) -> np.ndarray | None:
    """Crop from per-episode LMDB keyed {frame_idx:08d}_{L|R}."""
    lmdb_path = Path(crops_dir) / f"{episode_id}.lmdb"
    if not lmdb_path.is_dir():
        return None
    return read_crop_from_lmdb(
        lmdb_path, f"{int(frame_idx):08d}_{hand_code}")


# ── L. Alignment ────────────────────────────────────────────────────────────

def umeyama_alignment(
    src: np.ndarray,
    tgt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid alignment (R, t): tgt ~= src @ R.T + t."""
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    src_c = src - src.mean(axis=0, keepdims=True)
    tgt_c = tgt - tgt.mean(axis=0, keepdims=True)
    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign @ U.T
    t = tgt.mean(axis=0) - (R @ src.mean(axis=0))
    return R, t
