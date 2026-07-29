"""Offline visualization: project GT MANO meshes (both hands) onto EgoEMG webcam frames.

Supports two render modes:
  - mesh (default): pyrender shaded mesh with z-buffer occlusion
  - wireframe: OpenCV line-based wireframe overlay

Also saves per-frame outputs under frame_XXXXXXXX/:
  - rendered.png                    : mesh/wireframe overlay on frame
  - markers.png                     : mocap markers projected with skeleton
  - mano_left.glb, mano_right.glb  : MANO GT mesh in world space
  - fk_left.glb, fk_right.glb      : UmeTrack FK mesh in world space
  - occlusion.json                  : per-hand self-occlusion metrics
  - occlusion_vis.png               : vertices coloured by visibility (green=visible, red=occluded)

Usage:
    PYOPENGL_PLATFORM=osmesa python scripts/viz/visualize_egoemg_mesh.py \
        --memmap_dir data/EgoEMG_memmap \
        --data_root data/EgoEMG \
        --output /tmp/egoemg_mesh_samples \
        --n_samples 10
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from pathlib import Path

import cv2
import numpy as np
import pyrender
import smplx
import torch
import trimesh
from tqdm import tqdm

from emg2pose.occlusion import compute_self_occlusion
from emg2pose.UmeTrack.lib.common.hand import HandModel
from emg2pose.UmeTrack.lib.common.hand_skinning import _skin_points
from emg2pose.UmeTrack.lib.tracker.video_pose_data import load_hand_model_from_dict

HANDS = ["left", "right"]
HAND_COLORS_BGR = {
    "right": (0, 180, 255),  # Orange
    "left": (255, 180, 0),   # Blue
}
HAND_COLORS_RGB = {
    "right": (255, 180, 0),
    "left": (0, 180, 255),
}
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

SKELETON_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

# ── UmeTrack FK mesh helpers ─────────────────────────────────────────────────

_UMETRACK_HAND_MODEL = None


def _get_umetrack_hand_model() -> HandModel:
    global _UMETRACK_HAND_MODEL
    if _UMETRACK_HAND_MODEL is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "emg2pose" / "UmeTrack" / "dataset" / "generic_hand_model.json"
        )
        with open(path) as f:
            _UMETRACK_HAND_MODEL = load_hand_model_from_dict(json.load(f))
    return _UMETRACK_HAND_MODEL


def _mirror_hand_model(profile: HandModel) -> HandModel:
    mirrored_joint_rotation_axes = profile.joint_rotation_axes.clone()
    mirrored_joint_rest_positions = profile.joint_rest_positions.clone()
    mirrored_mesh_vertices = (
        profile.mesh_vertices.clone()
        if profile.mesh_vertices is not None
        else None
    )
    mirrored_joint_rotation_axes[..., 1:] *= -1
    mirrored_joint_rest_positions[..., 0] *= -1
    if mirrored_mesh_vertices is not None:
        mirrored_mesh_vertices[..., 0] *= -1
    return profile._replace(
        joint_rotation_axes=mirrored_joint_rotation_axes,
        joint_rest_positions=mirrored_joint_rest_positions,
        mesh_vertices=mirrored_mesh_vertices,
    )


def skin_fk_mesh(joint_angles: np.ndarray, flip: bool = False):
    """FK mesh skinning via UmeTrack hand model. Returns (verts, faces)."""
    profile = _get_umetrack_hand_model()
    if flip:
        profile = _mirror_hand_model(profile)
    ja_t = torch.from_numpy(np.asarray(joint_angles).copy()).float()
    leading_dims = ja_t.shape[:-1]
    wrist_transforms = torch.broadcast_to(torch.eye(4), leading_dims + (4, 4))
    vertices = _skin_points(
        profile.joint_rest_positions,
        profile.joint_rotation_axes,
        profile.dense_bone_weights,
        ja_t,
        profile.mesh_vertices,
        wrist_transforms,
    )
    vertices = vertices.reshape(list(leading_dims) + list(vertices.shape[-2:]))
    return vertices.cpu().numpy(), profile.mesh_triangles.cpu().numpy()


def save_mesh_glb(vertices_world: np.ndarray, faces: np.ndarray,
                  color: tuple[int, int, int], path: str) -> None:
    """Save world-space mesh as GLB with given vertex color."""
    if vertices_world is None or not np.isfinite(vertices_world).all():
        mesh = trimesh.creation.icosphere(radius=0.001)
        mesh.export(path)
        return
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)
    mesh.visual.vertex_colors = list(color) + [255]
    mesh.export(path)


def load_mm(manifest: dict, mm_dir: str, name: str) -> np.memmap:
    info = manifest["fields"][name]
    return np.memmap(
        f"{mm_dir}/{info['filename']}",
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def load_episode_mm(manifest: dict, mm_dir: str, name: str) -> np.memmap:
    info = manifest["episode_fields"][name]
    return np.memmap(
        f"{mm_dir}/{info['filename']}",
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def decode_bytes(values: np.ndarray) -> list[str]:
    return [
        v.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(v, (bytes, np.bytes_))
        else str(v)
        for v in values
    ]


def resolve_reencoded_video_path(
    raw_video_path: str,
    data_root: Path,
    allintra_root: Path,
    suffix: str,
) -> Path:
    raw_path = Path(raw_video_path)
    if raw_path.is_absolute():
        try:
            rel_path = raw_path.relative_to(data_root.resolve())
        except ValueError:
            rel_path = Path(raw_path.name)
    else:
        rel_path = raw_path
    return allintra_root / rel_path.with_name(f"{rel_path.stem}{suffix}.mp4")


def detect_active_region(
    frame_bgr: np.ndarray, calib_w: int, calib_h: int,
) -> tuple[int, int]:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    col_mean = g.mean(axis=0)
    ids = np.where(col_mean > 2.0)[0]
    if len(ids) > 0:
        return int(ids[0]), int(ids[-1]) + 1
    video_h = frame_bgr.shape[0]
    active_w = int(round(video_h * (calib_w / float(calib_h))))
    x0 = (frame_bgr.shape[1] - active_w) // 2
    return x0, x0 + active_w


def compute_video_intrinsics(
    K: np.ndarray,
    calib_w: int,
    calib_h: int,
    active_x0: int,
    active_x1: int,
    video_w: int,
    video_h: int,
) -> np.ndarray:
    """Compute intrinsics matrix mapped to raw video pixel coordinates."""
    active_w = float(active_x1 - active_x0)
    K_vid = np.eye(3, dtype=np.float64)
    K_vid[0, 0] = K[0, 0] * active_w / calib_w
    K_vid[1, 1] = K[1, 1] * video_h / calib_h
    K_vid[0, 2] = K[0, 2] * active_w / calib_w + active_x0
    K_vid[1, 2] = K[1, 2] * video_h / calib_h
    return K_vid


def project_world_to_video(
    points_world: np.ndarray,
    T_W_C: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    calib_w: int,
    calib_h: int,
    active_x0: int,
    active_x1: int,
    video_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    T_C_W = np.linalg.inv(T_W_C)
    R_C_W = T_C_W[:3, :3]
    t_C_W = T_C_W[:3, 3].reshape(3, 1)

    p_cam = (R_C_W @ points_world.T + t_C_W).T
    depth_valid = p_cam[:, 2] > 1e-6

    rvec, _ = cv2.Rodrigues(R_C_W)
    proj, _ = cv2.projectPoints(
        points_world.astype(np.float64), rvec, t_C_W, K, dist,
    )
    proj = proj.reshape(-1, 2).astype(np.float64)

    active_w = float(active_x1 - active_x0)
    proj[:, 0] = (proj[:, 0] / calib_w) * active_w + active_x0
    proj[:, 1] = (proj[:, 1] / calib_h) * video_h

    return proj, depth_valid


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


def draw_skeleton_2d(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    valid: np.ndarray,
    color_bgr: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    """Draw skeleton edges and joints on image."""
    out = image_bgr.copy()
    valid_bool = np.asarray(valid, dtype=bool)
    for i0, i1 in SKELETON_EDGES:
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
        cv2.circle(out, center, 3, color_bgr, -1, lineType=cv2.LINE_AA)
    if valid_bool.any():
        cy, cx = pts[valid_bool].mean(axis=0).astype(np.int32)
        cv2.putText(out, label, (int(cx) + 10, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)
    return out


def render_mesh_overlay(
    frame_bgr: np.ndarray,
    hand_meshes: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    T_W_C_cv: np.ndarray,
    K_vid: np.ndarray,
    renderer: pyrender.OffscreenRenderer,
    alpha: float,
) -> np.ndarray:
    """Render shaded MANO meshes and composite onto frame.

    Args:
        hand_meshes: list of (verts_world, faces, color_rgb) per hand.
    """
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])

    for verts, faces, color_rgb in hand_meshes:
        tm = trimesh.Trimesh(
            vertices=verts.astype(np.float32), faces=faces,
        )
        tm.visual.vertex_colors = list(color_rgb) + [255]
        scene.add(pyrender.Mesh.from_trimesh(tm, smooth=True))

    T_W_C_gl = T_W_C_cv @ FLIP_YZ
    cam = pyrender.IntrinsicsCamera(
        fx=K_vid[0, 0], fy=K_vid[1, 1],
        cx=K_vid[0, 2], cy=K_vid[1, 2],
        znear=0.01, zfar=100.0,
    )
    scene.add(cam, pose=T_W_C_gl)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=T_W_C_gl)

    color_rgb, depth = renderer.render(scene)

    mask = depth > 0
    if not mask.any():
        return frame_bgr

    overlay_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    out = frame_bgr.copy()
    mask_3 = mask[:, :, None]
    out = np.where(
        mask_3,
        (alpha * overlay_bgr + (1.0 - alpha) * frame_bgr).astype(np.uint8),
        out,
    )
    return out


def get_mano_verts_local(
    mano: smplx.MANO,
    pose_aa: np.ndarray,
    beta: np.ndarray,
    device: str,
) -> np.ndarray:
    global_orient = torch.zeros(1, 3, dtype=torch.float32, device=device)
    hand_pose = torch.tensor(
        pose_aa[3:48], dtype=torch.float32, device=device,
    ).unsqueeze(0)
    betas = torch.tensor(beta, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out = mano(global_orient=global_orient, hand_pose=hand_pose, betas=betas)
    return out.vertices[0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap_dir", default="data/EgoEMG_memmap")
    parser.add_argument("--data_root", default="data/EgoEMG")
    parser.add_argument(
        "--allintra_root",
        default="data/EgoEMG_allintra",
    )
    parser.add_argument("--allintra_suffix", default="_allintra")
    parser.add_argument(
        "--mano_model_path",
        default="../WiLoR/mano_data/models",
    )
    parser.add_argument("--output", default="/tmp/egoemg_mesh_samples")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument(
        "--render_mode", default="mesh", choices=["wireframe", "mesh"],
    )
    parser.add_argument("--mesh_alpha", type=float, default=0.7)
    args = parser.parse_args()

    mm_dir = args.memmap_dir
    manifest = json.load(open(f"{mm_dir}/manifest.json"))
    md = np.load(f"{mm_dir}/metadata.npz", allow_pickle=False)

    cam_tracked_mm = load_mm(manifest, mm_dir, "mocap_webcam_tracked")
    cam_transform_mm = load_mm(manifest, mm_dir, "mocap_webcam_transform")
    frame_idx_mm = load_mm(manifest, mm_dir, "image_webcam_frame_index")
    ep_idx_mm = load_mm(manifest, mm_dir, "episode_index")

    video_paths = decode_bytes(md["episode_webcam_video_path"])

    ego_root = Path(args.data_root).resolve()
    allintra_root = Path(args.allintra_root).resolve()

    with open(ego_root / "reprojection_assets" / "GX010023_standard_calibration.json", "r") as f:
        calib = json.load(f)
    K_raw = np.asarray(calib["camera_matrix"], dtype=np.float64)
    dist_raw = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    calib_w = int(calib["image_width"])
    calib_h = int(calib["image_height"])

    mano_model = smplx.MANO(
        model_path=str(args.mano_model_path),
        is_rhand=True,
        flat_hand_mean=False,
        use_pca=False,
        num_pca_comps=45,
    )
    mano_model.to(args.device)
    faces_right = mano_model.faces.copy()
    faces_left = mano_model.faces[:, [0, 2, 1]].copy()
    hand_faces = {"right": faces_right, "left": faces_left}

    hand_data = {}
    for hand in HANDS:
        hand_data[hand] = {
            "pose": load_mm(manifest, mm_dir, f"generated_mano_{hand}_pose"),
            "world": load_mm(manifest, mm_dir, f"mocap_mano_{hand}_world_transform"),
            "beta": load_episode_mm(manifest, mm_dir, f"generated_mano_{hand}_beta"),
            "joint_angles": load_mm(manifest, mm_dir, f"generated_joint_angles_{hand}"),
            "keypoints": load_mm(manifest, mm_dir, f"mocap_{hand}_keypoints"),
            "keypoints_valid": load_mm(manifest, mm_dir, f"mocap_{hand}_valid"),
        }
    beta_idx_arr = md["episode_beta_idx"]

    rng = np.random.RandomState(args.seed)
    valid_indices = np.where(cam_tracked_mm == 1)[0]
    n_samples = min(args.n_samples, len(valid_indices))
    sampled_indices = sorted(rng.choice(valid_indices, size=n_samples, replace=False))

    ep_frame_map: dict[int, list[int]] = {}
    for global_i in sampled_indices:
        ep = int(ep_idx_mm[global_i])
        ep_frame_map.setdefault(ep, []).append(global_i)

    from decord import VideoReader, cpu

    vrs: dict[int, VideoReader] = {}
    active_regions: dict[int, tuple[int, int]] = {}
    for ep in ep_frame_map:
        vp = resolve_reencoded_video_path(
            video_paths[ep], ego_root, allintra_root, args.allintra_suffix,
        )
        try:
            vr = VideoReader(str(vp), ctx=cpu(0))
            vrs[ep] = vr
            first_rgb = vr[0].asnumpy()
            first_bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
            active_regions[ep] = detect_active_region(first_bgr, calib_w, calib_h)
        except Exception:
            continue

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = None
    renderer_size = None
    pbar = tqdm(total=n_samples, desc="Rendering", unit="frame")

    for ep in sorted(vrs.keys()):
        vr = vrs[ep]
        x0, x1 = active_regions[ep]

        for global_i in sorted(ep_frame_map[ep]):
            device_frame_idx = int(frame_idx_mm[global_i])
            video_frame_idx = max(0, min(device_frame_idx, len(vr) - 1))

            try:
                frame_rgb = vr[video_frame_idx].asnumpy()
            except Exception:
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_h, video_w = frame_bgr.shape[:2]

            t12 = np.asarray(cam_transform_mm[global_i], dtype=np.float64)
            T_W_C = np.eye(4, dtype=np.float64)
            T_W_C[:3, :3] = t12[:9].reshape(3, 3)
            T_W_C[:3, 3] = t12[9:12]

            # Build world-space verts for both hands + FK mesh
            hand_world_verts: dict[str, np.ndarray] = {}
            hand_mano_faces: dict[str, np.ndarray] = {}
            fk_world_verts: dict[str, np.ndarray] = {}
            fk_faces: dict[str, np.ndarray] = {}
            for hand in HANDS:
                mano_pose = np.asarray(hand_data[hand]["pose"][global_i], dtype=np.float64)
                beta_idx = int(beta_idx_arr[ep])
                beta = np.asarray(hand_data[hand]["beta"][beta_idx], dtype=np.float64)

                verts_local = get_mano_verts_local(mano_model, mano_pose, beta, args.device)
                if hand == "left":
                    verts_local[:, 0] *= -1.0

                t12_world = np.asarray(hand_data[hand]["world"][global_i], dtype=np.float64)
                R_world = t12_world[:9].reshape(3, 3)
                t_world = t12_world[9:12]
                hand_world_verts[hand] = (R_world @ verts_local.T).T + t_world
                hand_mano_faces[hand] = hand_faces[hand]

                # FK mesh from UmeTrack joint angles.
                # Same convention as MANO: always skin a right hand,
                # then x-flip for left hand. FK faces need a winding flip
                # for both hands to match MANO/trimesh convention.
                ja = np.asarray(hand_data[hand]["joint_angles"][global_i], dtype=np.float32)
                if np.isfinite(ja).all() and np.abs(ja).sum() > 0:
                    try:
                        fk_v_local, fk_f = skin_fk_mesh(
                            joint_angles=ja[:20], flip=False,
                        )
                        fk_v_local = fk_v_local.copy()
                        if hand == "right":
                            fk_v_local[:, 0] *= -1.0
                            fk_f = fk_f[:, [0, 2, 1]].copy()
                        fk_span = np.median(
                            fk_v_local.max(axis=0) - fk_v_local.min(axis=0)
                        )
                        if fk_span > 1e-6:
                            fk_v_local = fk_v_local * (0.09 / fk_span)
                        fk_world_verts[hand] = (
                            R_world @ fk_v_local.T
                        ).T + t_world
                        fk_faces[hand] = fk_f
                    except Exception:
                        pass

            # Save GLB meshes in per-frame subdirectory
            frame_dir = out_dir / f"frame_{global_i:08d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for hand in HANDS:
                if hand in hand_world_verts:
                    save_mesh_glb(
                        hand_world_verts[hand], hand_mano_faces[hand],
                        HAND_COLORS_RGB[hand],
                        str(frame_dir / f"mano_{hand}.glb"),
                    )
                if hand in fk_world_verts:
                    save_mesh_glb(
                        fk_world_verts[hand], fk_faces[hand],
                        HAND_COLORS_RGB[hand],
                        str(frame_dir / f"fk_{hand}.glb"),
                    )

            # ── Self-occlusion analysis ─────────────────────────────────
            K_vid = compute_video_intrinsics(
                K_raw, calib_w, calib_h, x0, x1, video_w, video_h,
            )
            T_C_W = np.linalg.inv(T_W_C)
            R_C_W = T_C_W[:3, :3].astype(np.float64)
            t_C_W = T_C_W[:3, 3].astype(np.float64)

            occlusion_results: dict[str, dict] = {}
            for hand in HANDS:
                if hand not in hand_world_verts:
                    continue
                verts_w = hand_world_verts[hand].astype(np.float64)
                verts_cam = (R_C_W @ verts_w.T).T + t_C_W
                faces = hand_mano_faces[hand]

                result = compute_self_occlusion(
                    verts_cam, faces, K_vid, video_h, video_w,
                    depth_eps=0.005, window_half=2,
                )
                occlusion_results[hand] = result

            # Save per-hand metrics as JSON
            occ_json: dict[str, dict] = {}
            for hand, r in occlusion_results.items():
                occ_json[hand] = {
                    "occlusion_score": round(float(r["occlusion_score"]), 6),
                    "visible_ratio": round(float(r["visible_ratio"]), 6),
                    "n_visible": int(r["visible"].sum()),
                    "n_total": int(len(r["visible"])),
                    "area_weight_total": round(float(r["area_weights"].sum()), 6),
                }
            with open(str(frame_dir / "occlusion.json"), "w") as f:
                json.dump(occ_json, f, indent=2)

            # Visualisation: green = visible, red = occluded
            occ_vis = frame_bgr.copy()
            for hand, r in occlusion_results.items():
                for i in range(len(r["visible"])):
                    u, v = int(r["u_proj"][i]), int(r["v_proj"][i])
                    if 0 <= u < video_w and 0 <= v < video_h:
                        color = (0, 255, 0) if r["visible"][i] else (0, 0, 255)
                        cv2.circle(occ_vis, (u, v), 2, color, -1, lineType=cv2.LINE_AA)
            cv2.imwrite(str(frame_dir / "occlusion_vis.png"), occ_vis)

            # Marker projection + skeleton on a clean frame copy
            markers_bgr = frame_bgr.copy()
            for hand in HANDS:
                kp_world = np.asarray(
                    hand_data[hand]["keypoints"][global_i], dtype=np.float64,
                )
                kp_valid_raw = np.asarray(
                    hand_data[hand]["keypoints_valid"][global_i], dtype=bool,
                )
                if not kp_valid_raw.any():
                    continue
                kp_px, depth_valid = project_world_to_video(
                    kp_world, T_W_C, K_raw, dist_raw,
                    calib_w, calib_h, x0, x1, video_h,
                )
                in_image = (
                    (kp_px[:, 0] >= 0) & (kp_px[:, 0] < video_w)
                    & (kp_px[:, 1] >= 0) & (kp_px[:, 1] < video_h)
                )
                valid = depth_valid & in_image & kp_valid_raw & np.isfinite(kp_world).all(axis=1)
                if valid.sum() > 0:
                    markers_bgr = draw_skeleton_2d(
                        markers_bgr, kp_px, valid,
                        HAND_COLORS_BGR[hand], hand[0].upper(),
                    )
            cv2.imwrite(str(frame_dir / "markers.png"), markers_bgr)

            if args.render_mode == "mesh":
                frame_undist = cv2.undistort(frame_bgr, K_vid, dist_raw, None, K_vid)
                if (video_w, video_h) != renderer_size:
                    if renderer is not None:
                        renderer.delete()
                    renderer = pyrender.OffscreenRenderer(video_w, video_h)
                    renderer_size = (video_w, video_h)
                meshes = [
                    (hand_world_verts[h], hand_faces[h], HAND_COLORS_RGB[h])
                    for h in HANDS
                ]
                frame_bgr = render_mesh_overlay(
                    frame_undist, meshes, T_W_C, K_vid,
                    renderer, args.mesh_alpha,
                )
            else:
                for hand in HANDS:
                    verts_px, depth_valid = project_world_to_video(
                        hand_world_verts[hand], T_W_C, K_raw, dist_raw,
                        calib_w, calib_h, x0, x1, video_h,
                    )
                    in_image = (
                        (verts_px[:, 0] >= 0) & (verts_px[:, 0] < video_w)
                        & (verts_px[:, 1] >= 0) & (verts_px[:, 1] < video_h)
                    )
                    valid = depth_valid & in_image
                    draw_wireframe(
                        frame_bgr, verts_px, valid, hand_faces[hand],
                        HAND_COLORS_BGR[hand], args.line_width,
                    )

            y = 28
            for line in [f"global={global_i} ep={ep}", "R: orange  L: blue"]:
                cv2.putText(frame_bgr, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (20, 20, 20), 1, cv2.LINE_AA)
                y += 30

            cv2.imwrite(str(frame_dir / "rendered.png"), frame_bgr)
            pbar.update(1)

    pbar.close()
    if renderer is not None:
        renderer.delete()
    print(f"Done. Saved {n_samples} frames to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
