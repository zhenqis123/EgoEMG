"""Verify EgoEMG training data + EMG-only prediction visualization.

Samples EgoEMG training frames and overlays mocap/MANO references plus an
EMG-predicted hand pose for each sample.
For each hand sample, saves all verify_training outputs plus:
  emg_pred_original.png    - EMG→pose FK mesh projected on original frame
  emg_pred_crop.png         - EMG→pose FK mesh projected on crop
  emg_pred.glb              - EMG→pose FK mesh in world space (3D)
  comparison.png            - side-by-side: MANO GT (left) vs EMG pred (right)

Usage:
    python scripts/prepare/verify_training_with_emg.py --num-samples 20 --output-dir ./emg_verification
    python scripts/prepare/verify_training_with_emg.py --split test --num-samples 20
    python scripts/prepare/verify_training_with_emg.py --emg-ckpt /path/to/other.ckpt
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
DEFAULT_MEMMAP_DIR = Path("data/EgoEMG_v2_memmap")
DEFAULT_DATA_ROOT = Path("./data/EgoEMG")
DEFAULT_VIDEO_ROOT = Path("data/EgoEMG_videos")
MANO_MODEL_PATH = str(
    Path(os.environ.get("EGOEMG_ROOT", ".")) / "data" / "mano_data" / "models"
)
DEFAULT_EMG_CKPT = (
    "./logs/2026-04-30/23-28-41_emg2pose/"
    "regression_emgformer_small_aggressive_egoemg/version_0/checkpoints/"
    "egoemg-small-epoch=007-val_mae=0.2625.ckpt"
)
DEFAULT_NORM_STATS = "./assets/per_dataset_norm_stats.json"
# Default channel mapping matching config: emg2pose_interpolate16
DEFAULT_CHANNEL_INDICES_1BASED = [10, 12, 0, 1, 2, 4, 5, 6]
EMG_WINDOW_LENGTH = 7790

if str(DEFAULT_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_DATA_ROOT))

from reproject_hand_keypoints import (
    _map_processed_points_to_raw,
    _project_world_points,
    build_intrinsics_and_frame_mapper,
)

from egoemg.kinematics import forward_kinematics, load_default_hand_model

# ── FK mesh (UmeTrack skinning) ────────────────────────────────────────────
from egoemg.UmeTrack.lib.common.hand import HandModel
from egoemg.UmeTrack.lib.common.hand_skinning import _skin_points
from egoemg.UmeTrack.lib.tracker.video_pose_data import load_hand_model_from_dict
import smplx


def _load_umetrack_hand_model(filepath: str) -> HandModel:
    with open(filepath) as f:
        hand_model_dict = json.load(f)
    return load_hand_model_from_dict(hand_model_dict)


_UMETRACK_HAND_MODEL = None


def _get_umetrack_hand_model() -> HandModel:
    global _UMETRACK_HAND_MODEL
    if _UMETRACK_HAND_MODEL is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "emg2pose" / "UmeTrack" / "dataset" / "generic_hand_model.json"
        )
        _UMETRACK_HAND_MODEL = _load_umetrack_hand_model(str(path))
    return _UMETRACK_HAND_MODEL


def _mirror_hand_model(profile: HandModel) -> HandModel:
    mirrored_joint_rotation_axes = profile.joint_rotation_axes.clone()
    mirrored_joint_rest_positions = profile.joint_rest_positions.clone()
    mirrored_mesh_vertices = (
        profile.mesh_vertices.clone() if profile.mesh_vertices is not None else None
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
    user_profile = _get_umetrack_hand_model()
    if flip:
        user_profile = _mirror_hand_model(user_profile)
    ja_t = torch.from_numpy(np.asarray(joint_angles)).float()
    leading_dims = ja_t.shape[:-1]
    wrist_transforms = torch.broadcast_to(torch.eye(4), leading_dims + (4, 4))
    vertices = _skin_points(
        user_profile.joint_rest_positions,
        user_profile.joint_rotation_axes,
        user_profile.dense_bone_weights,
        ja_t,
        user_profile.mesh_vertices,
        wrist_transforms,
    )
    vertices = vertices.reshape(list(leading_dims) + list(vertices.shape[-2:]))
    triangles = user_profile.mesh_triangles
    return vertices.cpu().numpy(), triangles.cpu().numpy()


# ── Constants ────────────────────────────────────────────────────────────────

SKELETON_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

MIRROR_X_3 = np.array([-1.0, 1.0, 1.0], dtype=np.float32)

# ── Caches ──────────────────────────────────────────────────────────────────

_MEM_CACHE: dict = {}
_MANIFEST = None
_MD = None
_FK_HAND_MODEL = None
_EMG_MODEL = None
_EMG_NORM_STATS = None


def _manifest():
    global _MANIFEST
    if _MANIFEST is None:
        with open(_MEMMAP_DIR / "manifest.json") as f:
            _MANIFEST = json.load(f)
    return _MANIFEST


def _metadata():
    global _MD
    if _MD is None:
        _MD = dict(np.load(_MEMMAP_DIR / "metadata.npz", allow_pickle=False))
    return _MD


def _load_mm(name: str) -> np.memmap:
    if name not in _MEM_CACHE:
        mf = _manifest()
        info = mf["fields"][name]
        _MEM_CACHE[name] = np.memmap(
            _MEMMAP_DIR / info["filename"],
            dtype=np.dtype(info["dtype"]),
            mode="r",
            shape=tuple(info["shape"]),
        )
    return _MEM_CACHE[name]


def _get_fk_hand_model():
    global _FK_HAND_MODEL
    if _FK_HAND_MODEL is None:
        _FK_HAND_MODEL = load_default_hand_model()
    return _FK_HAND_MODEL


def _decode_str(val) -> str:
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8").rstrip("\x00")
    return str(val)


# ── EMG model ──────────────────────────────────────────────────────────────

def _load_emg_model(ckpt_path: str, device: str):
    """Load Emg2PoseFormer from a Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module_conf = ckpt["hyper_parameters"]["module_conf"]

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Build the module from its config
    conf = OmegaConf.create(module_conf)
    model = instantiate(conf).to(device)

    # Load weights (strip "model." prefix from lightning state_dict)
    state_dict = ckpt["state_dict"]
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


def _load_emg_norm_stats(norm_stats_path: str):
    """Load per-dataset normalization stats, return the egoemg entry."""
    with open(norm_stats_path) as f:
        stats = json.load(f)
    for key in stats:
        if "egoemg" in key.lower() and "qwerty" not in key.lower():
            return stats[key]
    return list(stats.values())[0]


def _convert_emg_to_16ch(emg_8ch: np.ndarray, channel_indices: list) -> np.ndarray:
    """Convert 8-channel EMG to 16-channel emg2pose layout (like dataset does)."""
    T = emg_8ch.shape[0]
    emg_16 = np.zeros((T, 16), dtype=np.float32)
    for i, ch_idx in enumerate(channel_indices):
        if 1 <= ch_idx <= 8:
            emg_16[:, i] = emg_8ch[:, ch_idx - 1]
    return emg_16


def predict_emg_center_frame(
    model, emg_window: np.ndarray, norm_stats: dict, device: str,
) -> np.ndarray:
    """Run EMG model on a full window, return center-frame joint angles (22,).

    emg_window: (T, 8) raw 8-channel EMG
    Returns: (22,) joint_angles in radians
    """
    # Convert 8ch → 16ch
    emg_16 = _convert_emg_to_16ch(emg_window, _CHANNEL_INDICES)
    # Normalize (scalar stats: per-channel mean/std)
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)
    emg_norm = (emg_16 - mean) / (std + 1e-6)
    # (T, 16) → (1, 16, T)
    emg_t = torch.from_numpy(emg_norm.T).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model({"emg": emg_t})  # (1, 22, T')
    # Take center frame
    center = preds.shape[-1] // 2
    ja = preds[0, :, center].cpu().numpy()  # (22,)
    return ja


# ── Projection ──────────────────────────────────────────────────────────────

def build_intrinsics_for_frame(frame_bgr, K_raw, dist_raw, calib_w, calib_h):
    video_h, video_w = frame_bgr.shape[:2]
    K_use, dist_use, intrinsics_info, _ = build_intrinsics_and_frame_mapper(
        K_raw, dist_raw, calib_w, calib_h, video_w, video_h,
        mode="gopro_8x7_crop_upsample", first_frame=frame_bgr,
    )
    return K_use, dist_use, intrinsics_info, video_w, video_h


def project_points(pts_world, T_W_C, K, dist, info):
    pts_proc, depth_valid = _project_world_points(pts_world, T_W_C, K, dist)
    pts_raw = _map_processed_points_to_raw(pts_proc, info)
    return pts_raw, depth_valid


# ── Drawing ─────────────────────────────────────────────────────────────────

def draw_skeleton_2d(img_bgr, pts, valid, color, label):
    valid = np.asarray(valid, dtype=bool)
    for i0, i1 in SKELETON_EDGES:
        if i0 >= len(pts) or i1 >= len(pts):
            continue
        if valid[i0] and valid[i1]:
            p0 = tuple(np.round(pts[i0]).astype(np.int32))
            p1 = tuple(np.round(pts[i1]).astype(np.int32))
            cv2.line(img_bgr, p0, p1, color, 2, lineType=cv2.LINE_AA)
    for i, (p, v) in enumerate(zip(pts, valid)):
        if not v:
            continue
        center = tuple(np.round(p).astype(np.int32))
        cv2.circle(img_bgr, center, 3, color, -1, lineType=cv2.LINE_AA)
    if valid.any():
        cy, cx = pts[valid].mean(axis=0).astype(np.int32)
        cv2.putText(img_bgr, label, (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img_bgr


def draw_wireframe(image_bgr, points_xy, valid, faces, color_bgr, line_width=1):
    out = image_bgr.copy()
    valid = np.asarray(valid, dtype=bool)
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        p0 = tuple(np.round(points_xy[i0]).astype(np.int32))
        p1 = tuple(np.round(points_xy[i1]).astype(np.int32))
        p2 = tuple(np.round(points_xy[i2]).astype(np.int32))
        cv2.line(out, p0, p1, color_bgr, line_width, lineType=cv2.LINE_AA)
        cv2.line(out, p1, p2, color_bgr, line_width, lineType=cv2.LINE_AA)
        cv2.line(out, p2, p0, color_bgr, line_width, lineType=cv2.LINE_AA)
    return out


def get_points_bbox(pts, valid, img_w, img_h, margin=20):
    v = np.asarray(valid, dtype=bool)
    valid_pts = pts[v]
    if len(valid_pts) < 2:
        return None
    xmin = int(np.floor(valid_pts[:, 0].min()))
    xmax = int(np.ceil(valid_pts[:, 0].max()))
    ymin = int(np.floor(valid_pts[:, 1].min()))
    ymax = int(np.ceil(valid_pts[:, 1].max()))
    sz = max(xmax - xmin, ymax - ymin) // 2 + margin
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    x0 = max(0, cx - sz)
    x1 = min(img_w, cx + sz)
    y0 = max(0, cy - sz)
    y1 = min(img_h, cy + sz)
    return int(x0), int(y0), int(x1), int(y1)


def crop_hand_region(frame_bgr, bbox, target_size=256):
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    return img


def make_comparison_image(img_gt, img_emg, label_gt="MANO GT", label_emg="EMG Pred"):
    """Create side-by-side comparison: GT (left) | EMG pred (right)."""
    h = max(img_gt.shape[0], img_emg.shape[0])
    w = img_gt.shape[1] + img_emg.shape[1] + 4
    canvas = np.zeros((h + 30, w, 3), dtype=np.uint8)
    canvas[:img_gt.shape[0], :img_gt.shape[1]] = img_gt
    x_off = img_gt.shape[1] + 4
    canvas[:img_emg.shape[0], x_off:x_off + img_emg.shape[1]] = img_emg
    cv2.putText(canvas, label_gt, (10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, label_emg, (x_off + 10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# ── GLB export ──────────────────────────────────────────────────────────────

def save_markers_glb(markers, valid, path):
    valid = np.asarray(valid, dtype=bool)
    pts = markers[valid]
    if len(pts) < 2:
        mesh = trimesh.creation.icosphere(radius=0.001)
        mesh.export(str(path))
        return
    spheres = []
    for pt in pts:
        s = trimesh.creation.icosphere(radius=0.003, subdivisions=2)
        s.apply_translation(pt)
        s.visual.vertex_colors = [255, 100, 100, 255]
        spheres.append(s)
    scene = trimesh.util.concatenate(spheres)
    scene.export(str(path))


def save_fk_mesh_glb(vertices_world, faces, path, color=None):
    if vertices_world is None or not np.isfinite(vertices_world).all():
        mesh = trimesh.creation.icosphere(radius=0.001)
        mesh.export(str(path))
        return
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)
    if color is not None:
        mesh.visual.vertex_colors = color
    else:
        mesh.visual.vertex_colors = [100, 180, 100, 255]
    mesh.export(str(path))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify training data + EMG only prediction visualization"
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="./emg_verification_samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crops-dir", default=None,
                        help="Path to per-episode LMDB crops")
    parser.add_argument("--split", default="both",
                        choices=["train", "test", "both"],
                        help="Which data split to sample from")
    parser.add_argument("--emg-ckpt", default=DEFAULT_EMG_CKPT,
                        help="Path to EMG model checkpoint")
    parser.add_argument("--memmap-dir", default=str(DEFAULT_MEMMAP_DIR),
                        help="Path to EgoEMG memmap directory")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--norm-stats", default=DEFAULT_NORM_STATS)
    parser.add_argument("--channel-indices", default=None,
                        help="Comma-separated 0-based 16-position EMG channel indices (e.g. 10,12,0,1,2,4,5,6)")
    args = parser.parse_args()

    # Set globals used by helper functions
    global _MEMMAP_DIR, _CHANNEL_INDICES
    _MEMMAP_DIR = Path(args.memmap_dir)
    if args.channel_indices:
        _CHANNEL_INDICES = [int(x) for x in args.channel_indices.split(",")]
    else:
        _CHANNEL_INDICES = DEFAULT_CHANNEL_INDICES_1BASED

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load EMG model ────────────────────────────────────────────────────
    print(f"Loading EMG model from {args.emg_ckpt} ...")
    emg_model = _load_emg_model(args.emg_ckpt, device)
    emg_norm_stats = _load_emg_norm_stats(args.norm_stats)
    print("  Model loaded.")

    # Load metadata
    md = _metadata()
    ep_ids = [_decode_str(x) for x in md["episode_id"]]
    ep_starts = np.asarray(md["episode_start_idx"], dtype=np.int64)
    ep_ends = np.asarray(md["episode_end_idx"], dtype=np.int64)
    total_rows = len(_load_mm("episode_index"))

    # Load memmaps
    wf_idx_mm = _load_mm("image_webcam_frame_index")
    stale_mm = _load_mm("image_webcam_stale")
    tracked_mm = _load_mm("mocap_webcam_tracked")
    video_rel_paths = [_decode_str(x) for x in md["episode_webcam_video_path"]]

    # Splits
    split_id_mm = np.memmap(
        _MEMMAP_DIR / "frame_split_id.dat",
        dtype=np.int8, mode="r",
    )
    if args.split == "train":
        allowed_splits = {0}
    elif args.split == "test":
        allowed_splits = {1, 2, 3}  # user, gesture, both
    else:
        allowed_splits = None

    # MANO betas
    try:
        beta_left_mm = np.memmap(
            _MEMMAP_DIR / "generated_mano_left_beta.dat",
            dtype=np.float32, mode="r", shape=(41, 10),
        )
        beta_right_mm = np.memmap(
            _MEMMAP_DIR / "generated_mano_right_beta.dat",
            dtype=np.float32, mode="r", shape=(41, 10),
        )
        HAVE_BETAS = True
    except Exception:
        print("Warning: MANO beta files not found, using zeros")
        HAVE_BETAS = False

    # Init MANO layer
    mano_layer = smplx.MANO(
        model_path=MANO_MODEL_PATH, is_rhand=True,
        flat_hand_mean=False, use_pca=False, num_pca_comps=45,
    ).to(device)
    mano_faces = mano_layer.faces.copy()

    # Init FK hand model
    fk_hand_model = _get_fk_hand_model()

    # Load calibration
    calib_path = Path(args.data_root) / "reprojection_assets" / "GX010023_standard_calibration.json"
    with open(calib_path) as f:
        calib = json.load(f)
    K_raw = np.asarray(calib["camera_matrix"], dtype=np.float64)
    dist_raw = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    calib_w = int(calib["image_width"])
    calib_h = int(calib["image_height"])

    # Sample candidates
    all_candidates = []
    for ep_idx in range(len(ep_ids)):
        start = int(ep_starts[ep_idx])
        end = int(ep_ends[ep_idx])
        n = end - start
        n_samples = min(30, max(2, int(n / total_rows * args.num_samples * 5)))
        positions = rng.integers(start, end, size=n_samples)
        valid_left_mm = _load_mm("mocap_left_valid")
        valid_right_mm = _load_mm("mocap_right_valid")
        for pos in positions:
            # Filter by split
            if allowed_splits is not None and int(split_id_mm[pos]) not in allowed_splits:
                continue
            # Need enough EMG context on both sides
            half_win = EMG_WINDOW_LENGTH // 2
            if pos < half_win or pos >= len(split_id_mm) - half_win:
                continue
            if bool(tracked_mm[pos]) and not bool(stale_mm[pos]):
                left_ok = bool(valid_left_mm[pos].any())
                right_ok = bool(valid_right_mm[pos].any())
                if left_ok or right_ok:
                    all_candidates.append((ep_idx, int(pos)))

    print(f"Total candidates (split={args.split}): {len(all_candidates)}")
    n_select = min(args.num_samples, len(all_candidates))
    if n_select == 0:
        print("No candidates found!")
        return
    selected = rng.choice(all_candidates, size=n_select, replace=False)

    # Group by episode
    by_ep = {}
    for ep_idx, abs_idx in selected:
        by_ep.setdefault(ep_idx, []).append(abs_idx)

    # Decord
    try:
        from decord import VideoReader, cpu as decord_cpu
        HAVE_DECORD = True
    except Exception:
        HAVE_DECORD = False
        print("Warning: decord not available, falling back to OpenCV")

    sample_idx = 0
    for ep_idx, abs_indices in tqdm(sorted(by_ep.items()), desc="Episodes", unit="ep"):
        raw_video_path = str(Path(args.video_root) / video_rel_paths[ep_idx])
        video_path = raw_video_path.replace(".mp4", "_allintra.mp4")
        if not os.path.exists(video_path):
            print(f"  Video not found: {video_path}, skip {ep_ids[ep_idx]}")
            continue

        if HAVE_DECORD:
            vr = VideoReader(video_path, ctx=decord_cpu(0))
            total_vf = len(vr)

            def read_frame(vfid):
                vfid = max(0, min(int(vfid), total_vf - 1))
                return cv2.cvtColor(vr[vfid].asnumpy(), cv2.COLOR_RGB2BGR)
        else:
            cap = cv2.VideoCapture(video_path)
            total_vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            def read_frame(vfid):
                vfid = max(0, min(int(vfid), total_vf - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, vfid)
                ret, frame = cap.read()
                return frame if ret else None

        if HAVE_BETAS:
            beta_left = np.asarray(beta_left_mm[ep_idx], dtype=np.float32)
            beta_right = np.asarray(beta_right_mm[ep_idx], dtype=np.float32)
        else:
            beta_left = np.zeros(10, dtype=np.float32)
            beta_right = np.zeros(10, dtype=np.float32)

        for abs_idx in tqdm(sorted(abs_indices), desc=f"  {ep_ids[ep_idx]}",
                            unit="f", leave=False):
            vfid = int(wf_idx_mm[abs_idx])
            vfid = max(0, min(vfid, total_vf - 1))
            frame_bgr = read_frame(vfid)
            if frame_bgr is None:
                continue
            frame_bgr = frame_bgr.copy()
            orig_clean = frame_bgr.copy()

            K_use, dist_use, intrinsics_info, video_w, video_h = \
                build_intrinsics_for_frame(frame_bgr, K_raw, dist_raw, calib_w, calib_h)

            cam_transform_mm = _load_mm("mocap_webcam_transform")
            t12 = np.asarray(cam_transform_mm[abs_idx], dtype=np.float64)
            T_W_C = np.eye(4, dtype=np.float64)
            T_W_C[:3, :3] = t12[:9].reshape(3, 3)
            T_W_C[:3, 3] = t12[9:12]

            for hand in ["left", "right"]:
                kp_mm = _load_mm(f"mocap_{hand}_keypoints")
                valid_mm = _load_mm(f"mocap_{hand}_valid")
                ja_mm = _load_mm(f"generated_joint_angles_{hand}")
                mano_pose_mm = _load_mm(f"generated_mano_{hand}_pose")
                mano_world_mm = _load_mm(f"mocap_mano_{hand}_world_transform")
                beta = beta_left if hand == "left" else beta_right

                kp_world = np.asarray(kp_mm[abs_idx], dtype=np.float64)
                valid_kp = np.asarray(valid_mm[abs_idx], dtype=bool)
                ja_20d = np.asarray(ja_mm[abs_idx], dtype=np.float32)
                mano_pose = np.asarray(mano_pose_mm[abs_idx], dtype=np.float32)

                # ── Marker projection ──
                pts_marker, depth_valid = project_points(
                    kp_world, T_W_C, K_use, dist_use, intrinsics_info,
                )
                valid_marker = (
                    depth_valid
                    & (pts_marker[:, 0] >= 0) & (pts_marker[:, 0] < video_w)
                    & (pts_marker[:, 1] >= 0) & (pts_marker[:, 1] < video_h)
                    & valid_kp
                    & np.isfinite(kp_world).all(axis=1)
                )

                # ── MANO mesh decode ──
                mano_verts_world = None
                mano_faces_hand = mano_faces.copy()
                if np.isfinite(mano_pose).all() and np.abs(mano_pose).sum() > 0:
                    try:
                        global_orient = torch.zeros(1, 3, dtype=torch.float32, device=device)
                        hand_pose_aa = mano_pose[3:48].astype(np.float32)
                        hp_t = torch.tensor(hand_pose_aa, dtype=torch.float32, device=device).unsqueeze(0)
                        betas_t = torch.tensor(beta, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            out = mano_layer(global_orient=global_orient, hand_pose=hp_t, betas=betas_t)
                        verts_local = out.vertices[0].cpu().numpy()
                        if hand == "left":
                            verts_local = verts_local * MIRROR_X_3
                            mano_faces_hand = mano_faces_hand[:, [0, 2, 1]]
                        t12_world = np.asarray(mano_world_mm[abs_idx], dtype=np.float64)
                        R_world = t12_world[:9].reshape(3, 3)
                        t_world = t12_world[9:12]
                        mano_verts_world = (R_world @ verts_local.T).T + t_world
                    except Exception as e:
                        tqdm.write(f"    MANO decode failed for {hand}: {e}")

                # ── MANO projection ──
                mano_pts_raw = None
                mano_valid = None
                if mano_verts_world is not None:
                    verts_proc, depth_valid_m = _project_world_points(
                        mano_verts_world, T_W_C, K_use, dist_use,
                    )
                    mano_pts_raw = _map_processed_points_to_raw(verts_proc, intrinsics_info)
                    mano_valid = (
                        depth_valid_m
                        & (mano_pts_raw[:, 0] >= 0) & (mano_pts_raw[:, 0] < video_w)
                        & (mano_pts_raw[:, 1] >= 0) & (mano_pts_raw[:, 1] < video_h)
                    )

                # ── FK from GT joint_angles ──
                fk_verts_world = None
                fk_faces = None
                if np.isfinite(ja_20d).all() and np.abs(ja_20d).sum() > 0:
                    try:
                        fk_verts_local, fk_faces = skin_fk_mesh(
                            joint_angles=ja_20d[:20], flip=(hand == "left"),
                        )
                        fk_verts_local = fk_verts_local.copy()
                        fk_span = np.median(
                            fk_verts_local.max(axis=0) - fk_verts_local.min(axis=0)
                        )
                        if fk_span > 1e-6:
                            fk_verts_local = fk_verts_local * (0.09 / fk_span)
                        t12_world = np.asarray(mano_world_mm[abs_idx], dtype=np.float64)
                        R_world_fk = t12_world[:9].reshape(3, 3)
                        t_world_fk = t12_world[9:12]
                        fk_verts_world = (
                            R_world_fk @ fk_verts_local.T
                        ).T + t_world_fk
                    except Exception as e:
                        tqdm.write(f"    FK mesh failed for {hand}: {e}")

                # ── EMG prediction ──
                emg_verts_world = None
                emg_faces = None
                emg_ja_full = None
                try:
                    emg_field = f"emg_{hand}_filtered"
                    emg_mm = _load_mm(emg_field)
                    half_win = EMG_WINDOW_LENGTH // 2
                    emg_start = abs_idx - half_win
                    emg_end = abs_idx + half_win
                    # Clamp to valid range
                    emg_start = max(0, emg_start)
                    emg_end = min(emg_mm.shape[0], emg_end)
                    # If window is too short, pad
                    emg_window = np.asarray(emg_mm[emg_start:emg_end], dtype=np.float32)
                    if emg_window.shape[0] < EMG_WINDOW_LENGTH:
                        padded = np.zeros((EMG_WINDOW_LENGTH, emg_window.shape[1]), dtype=np.float32)
                        offset = max(0, half_win - abs_idx)
                        actual_len = min(emg_window.shape[0], EMG_WINDOW_LENGTH - offset)
                        padded[offset:offset + actual_len] = emg_window[:actual_len]
                        emg_window = padded

                    emg_ja_full = predict_emg_center_frame(
                        emg_model, emg_window, emg_norm_stats, device,
                    )  # (22,) in radians

                    if np.isfinite(emg_ja_full).all():
                        emg_verts_local, emg_faces = skin_fk_mesh(
                            joint_angles=emg_ja_full[:20], flip=(hand == "left"),
                        )
                        emg_verts_local = emg_verts_local.copy()
                        fk_span = np.median(
                            emg_verts_local.max(axis=0) - emg_verts_local.min(axis=0)
                        )
                        if fk_span > 1e-6:
                            emg_verts_local = emg_verts_local * (0.09 / fk_span)
                        t12_world = np.asarray(mano_world_mm[abs_idx], dtype=np.float64)
                        R_world_emg = t12_world[:9].reshape(3, 3)
                        t_world_emg = t12_world[9:12]
                        emg_verts_world = (
                            R_world_emg @ emg_verts_local.T
                        ).T + t_world_emg
                except Exception as e:
                    tqdm.write(f"    EMG prediction failed for {hand}: {e}")

                # ── EMG projection ──
                emg_pts_raw = None
                emg_valid = None
                if emg_verts_world is not None:
                    verts_proc_e, depth_valid_e = _project_world_points(
                        emg_verts_world, T_W_C, K_use, dist_use,
                    )
                    emg_pts_raw = _map_processed_points_to_raw(verts_proc_e, intrinsics_info)
                    emg_valid = (
                        depth_valid_e
                        & (emg_pts_raw[:, 0] >= 0) & (emg_pts_raw[:, 0] < video_w)
                        & (emg_pts_raw[:, 1] >= 0) & (emg_pts_raw[:, 1] < video_h)
                    )

                # ── Bbox from markers ──
                bbox = get_points_bbox(pts_marker, valid_marker, video_w, video_h, margin=30)
                crop = crop_hand_region(frame_bgr, bbox, target_size=256)
                crop_clean = crop_hand_region(orig_clean, bbox, target_size=256)

                # ── Bbox from MANO verts ──
                mano_bbox = None
                if mano_pts_raw is not None and mano_valid is not None:
                    mano_bbox = get_points_bbox(mano_pts_raw, mano_valid, video_w, video_h, margin=20)

                # ── Draw markers on original ──
                markers_orig = orig_clean.copy()
                if valid_marker.sum() > 0:
                    markers_orig = draw_skeleton_2d(
                        markers_orig, pts_marker, valid_marker,
                        color=(0, 255, 255), label=hand[0].upper(),
                    )

                # ── Draw markers on crop ──
                markers_crop = None
                if crop is not None and bbox is not None and valid_marker.sum() > 0:
                    x0, y0, _, _ = bbox
                    pts_crop = pts_marker.copy()
                    pts_crop[:, 0] -= x0
                    pts_crop[:, 1] -= y0
                    scale = 256.0 / (bbox[2] - bbox[0])
                    pts_crop[:, 0] *= scale
                    pts_crop[:, 1] *= scale
                    valid_crop = valid_marker.copy()
                    in_crop = (
                        (pts_crop[:, 0] >= 0) & (pts_crop[:, 0] < 256)
                        & (pts_crop[:, 1] >= 0) & (pts_crop[:, 1] < 256)
                    )
                    valid_crop = valid_crop & in_crop
                    markers_crop = crop.copy()
                    if valid_crop.sum() > 0:
                        markers_crop = draw_skeleton_2d(
                            markers_crop, pts_crop, valid_crop,
                            color=(0, 255, 255), label=hand[0].upper(),
                        )

                # ── Draw MANO wireframe ──
                mano_orig = orig_clean.copy()
                if mano_pts_raw is not None and mano_valid is not None and mano_valid.sum() > 0:
                    mano_orig = draw_wireframe(mano_orig, mano_pts_raw, mano_valid,
                                               mano_faces_hand, color_bgr=(255, 180, 0))
                mano_crop = None
                if crop is not None and bbox is not None and \
                   mano_pts_raw is not None and mano_valid is not None and mano_valid.sum() > 0:
                    x0, y0, _, _ = bbox
                    m_crop = mano_pts_raw.copy()
                    m_crop[:, 0] -= x0
                    m_crop[:, 1] -= y0
                    scale = 256.0 / (bbox[2] - bbox[0])
                    m_crop[:, 0] *= scale
                    m_crop[:, 1] *= scale
                    in_crop = (
                        (m_crop[:, 0] >= 0) & (m_crop[:, 0] < 256)
                        & (m_crop[:, 1] >= 0) & (m_crop[:, 1] < 256)
                    )
                    m_valid_crop = mano_valid & in_crop
                    if m_valid_crop.sum() > 0:
                        mano_crop = draw_wireframe(
                            crop.copy(), m_crop, m_valid_crop,
                            mano_faces_hand, color_bgr=(255, 180, 0),
                        )

                # ── Draw EMG prediction wireframe (green) ──
                emg_orig = orig_clean.copy()
                if emg_pts_raw is not None and emg_valid is not None and emg_valid.sum() > 0:
                    emg_orig = draw_wireframe(emg_orig, emg_pts_raw, emg_valid,
                                              emg_faces, color_bgr=(0, 255, 100))
                    cv2.putText(emg_orig, "EMG", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2, cv2.LINE_AA)
                emg_crop = None
                if crop is not None and bbox is not None and \
                   emg_pts_raw is not None and emg_valid is not None and emg_valid.sum() > 0:
                    x0, y0, _, _ = bbox
                    e_crop = emg_pts_raw.copy()
                    e_crop[:, 0] -= x0
                    e_crop[:, 1] -= y0
                    scale = 256.0 / (bbox[2] - bbox[0])
                    e_crop[:, 0] *= scale
                    e_crop[:, 1] *= scale
                    in_crop = (
                        (e_crop[:, 0] >= 0) & (e_crop[:, 0] < 256)
                        & (e_crop[:, 1] >= 0) & (e_crop[:, 1] < 256)
                    )
                    e_valid_crop = emg_valid & in_crop
                    if e_valid_crop.sum() > 0:
                        emg_crop = draw_wireframe(
                            crop.copy(), e_crop, e_valid_crop,
                            emg_faces, color_bgr=(0, 255, 100),
                        )

                # ── Bbox on original ──
                bbox_orig = orig_clean.copy()
                if mano_bbox is not None:
                    bbox_orig = draw_bbox(bbox_orig, mano_bbox)

                # ── Comparison image ──
                comparison = None
                if mano_pts_raw is not None and mano_valid is not None and \
                   emg_pts_raw is not None and emg_valid is not None:
                    gt_mano = orig_clean.copy()
                    emg_only = orig_clean.copy()
                    if mano_valid.sum() > 0:
                        gt_mano = draw_wireframe(gt_mano, mano_pts_raw, mano_valid,
                                                 mano_faces_hand, color_bgr=(255, 180, 0))
                    if emg_valid.sum() > 0:
                        emg_only = draw_wireframe(emg_only, emg_pts_raw, emg_valid,
                                                  emg_faces, color_bgr=(0, 255, 100))
                    comparison = make_comparison_image(gt_mano, emg_only)

                # ── Create sample dir ──
                ep_id = ep_ids[ep_idx]
                sample_dir = Path(args.output_dir) / f"sample_{sample_idx:04d}_{ep_id}_{hand}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                # ── Save pre-crop from LMDB ──
                if args.crops_dir is not None:
                    try:
                        import lmdb
                        hand_code = "L" if hand == "left" else "R"
                        crop_lmdb = Path(args.crops_dir) / f"{ep_id}.lmdb"
                        if crop_lmdb.exists():
                            env = lmdb.open(str(crop_lmdb), readonly=True, lock=False,
                                            readahead=False)
                            with env.begin() as txn:
                                key = f"{vfid:08d}_{hand_code}".encode()
                                jpeg_bytes = txn.get(key)
                                if jpeg_bytes is not None:
                                    from PIL import Image
                                    precrop_img = np.asarray(Image.open(io.BytesIO(jpeg_bytes)))
                                    precrop_bgr = cv2.cvtColor(precrop_img, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(str(sample_dir / "precrop.png"), precrop_bgr)
                            env.close()
                    except Exception as e:
                        tqdm.write(f"    Pre-crop lookup failed for {hand}: {e}")

                # ── Save images ──
                cv2.imwrite(str(sample_dir / "original.png"), orig_clean)
                if crop_clean is not None:
                    cv2.imwrite(str(sample_dir / "crop.png"), crop_clean)
                cv2.imwrite(str(sample_dir / "markers_proj_original.png"), markers_orig)
                if markers_crop is not None:
                    cv2.imwrite(str(sample_dir / "markers_proj_crop.png"), markers_crop)
                cv2.imwrite(str(sample_dir / "mano_proj_original.png"), mano_orig)
                if mano_crop is not None:
                    cv2.imwrite(str(sample_dir / "mano_proj_crop.png"), mano_crop)
                cv2.imwrite(str(sample_dir / "mano_bbox_original.png"), bbox_orig)

                # ── Save EMG prediction overlay ──
                cv2.imwrite(str(sample_dir / "emg_pred_original.png"), emg_orig)
                if emg_crop is not None:
                    cv2.imwrite(str(sample_dir / "emg_pred_crop.png"), emg_crop)

                # ── Save comparison ──
                if comparison is not None:
                    cv2.imwrite(str(sample_dir / "comparison.png"), comparison)

                # ── Save GLBs ──
                if valid_kp.sum() > 1:
                    save_markers_glb(kp_world, valid_kp, str(sample_dir / "markers.glb"))

                if fk_verts_world is not None and fk_faces is not None:
                    save_fk_mesh_glb(fk_verts_world, fk_faces,
                                     str(sample_dir / "gt_from_angles.glb"))

                if mano_verts_world is not None:
                    mesh = trimesh.Trimesh(vertices=mano_verts_world, faces=mano_faces_hand)
                    mesh.visual.vertex_colors = [160, 180, 220, 255]
                    mesh.export(str(sample_dir / "mano_gt.glb"))

                if emg_verts_world is not None and emg_faces is not None:
                    save_fk_mesh_glb(
                        emg_verts_world, emg_faces,
                        str(sample_dir / "emg_pred.glb"),
                        color=[100, 255, 100, 255],
                    )

                sample_idx += 1

        if not HAVE_DECORD:
            cap.release()

    print(f"\nSaved {sample_idx} samples to {args.output_dir}/")
    print("Each sample dir contains:")
    for f in [
        "original.png", "crop.png",
        "markers_proj_original.png", "markers_proj_crop.png",
        "mano_proj_original.png", "mano_proj_crop.png",
        "mano_bbox_original.png",
        "emg_pred_original.png", "emg_pred_crop.png",
        "comparison.png",
        "markers.glb", "gt_from_angles.glb", "mano_gt.glb", "emg_pred.glb",
    ]:
        if args.crops_dir is not None:
            print(f"  {f}")
        elif f != "precrop.png":
            print(f"  {f}")


if __name__ == "__main__":
    main()
