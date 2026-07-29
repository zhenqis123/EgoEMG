"""Find test samples where fusion most outperforms vision-only, with rich visualizations.

Evaluates the fusion checkpoint, extracts both vision-only and fusion predictions
from a single forward pass, ranks samples by MAE improvement, and saves full
visualizations: original frame, hand crop, marker projection, MANO mesh overlay,
FK skeleton from predictions, per-joint error comparison, and 3D GLB exports.

Usage:
    python scripts/eval/find_fusion_wins.py \
        --config-name experiment/fusion/vision_resnet_small_emgfusion_center \
        --checkpoint logs/fusion/resnet_small_emgfusion_center/version_12/checkpoints/resnet-small-centerfusion-epoch=039-val_mae=0.0978.ckpt \
        --data-location data/EgoEMG_v2_memmap \
        --video-root data/EgoEMG_allintra \
        --num-samples 20 \
        --output-dir ./fusion_wins_viz
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── Headless matplotlib ──────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size": 9})

# ── Project imports ──────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from emg2pose.lightning import EmgPredictionModule
from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from emg2pose.kinematics import forward_kinematics, load_default_hand_model
from emg2pose.models.heads.mlp import MLPHead
from emg2pose.models.modules.emgformer import Emg2PoseFormer
from emg2pose.models.modules.vit_vision import VisionViTPose
from emg2pose.models.modules.resnet_vision import ResNetVisionPose

# ── UmeTrack / reprojection imports (matching verify_training_with_emg.py) ─────
_DATA_ROOT = Path("./data/EgoEMG")
if str(_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_ROOT))
from reproject_hand_keypoints import (
    _map_processed_points_to_raw,
    _project_world_points,
    build_intrinsics_and_frame_mapper,
)

# ── UmeTrack hand model for skinning ─────────────────────────────────────────
from emg2pose.UmeTrack.lib.common.hand import HandModel
from emg2pose.UmeTrack.lib.common.hand_skinning import _skin_points
from emg2pose.UmeTrack.lib.tracker.video_pose_data import load_hand_model_from_dict

import smplx
import trimesh

CONFIG_DIR = str(_PROJECT_DIR / "config")
MANO_MODEL_PATH = "../WiLoR/mano_data/models"
UMETRACK_HAND_MODEL_PATH = str(
    _PROJECT_DIR / "emg2pose" / "UmeTrack" / "dataset" / "generic_hand_model.json"
)

warnings.filterwarnings(
    "ignore", message="The given NumPy array is not writable", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="enable_nested_tensor is True", category=UserWarning
)

# ── Constants ────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "Th_CMC_F", "Th_CMC_A", "Th_MCP", "Th_IP",
    "Ix_MCP_F", "Ix_MCP_A", "Ix_PIP", "Ix_DIP",
    "Md_MCP_F", "Md_MCP_A", "Md_PIP", "Md_DIP",
    "Rg_MCP_F", "Rg_MCP_A", "Rg_PIP", "Rg_DIP",
    "Pk_MCP_F", "Pk_MCP_A", "Pk_PIP", "Pk_DIP",
    "Wr_FE", "Wr_RU",
]

SKELETON_EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

MIRROR_X_3 = np.array([-1.0, 1.0, 1.0], dtype=np.float32)

# ── Caches ───────────────────────────────────────────────────────────────────
_UMETRACK_HAND_MODEL: Optional[HandModel] = None
_FK_HAND_MODEL = None
_MANO_LAYER = None
_MANIFEST: Optional[dict] = None
_MM_CACHE: dict[str, np.memmap] = {}


def _load_manifest(memmap_dir: Path) -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        with open(memmap_dir / "manifest.json") as f:
            _MANIFEST = json.load(f)
    return _MANIFEST


def _load_mm(memmap_dir: Path, name: str) -> np.memmap:
    if name not in _MM_CACHE:
        mf = _load_manifest(memmap_dir)
        info = mf["fields"][name]
        _MM_CACHE[name] = np.memmap(
            memmap_dir / info["filename"],
            dtype=np.dtype(info["dtype"]),
            mode="r",
            shape=tuple(info["shape"]),
        )
    return _MM_CACHE[name]


def _get_umetrack_hand_model() -> HandModel:
    global _UMETRACK_HAND_MODEL
    if _UMETRACK_HAND_MODEL is None:
        with open(UMETRACK_HAND_MODEL_PATH) as f:
            hand_model_dict = json.load(f)
        _UMETRACK_HAND_MODEL = load_hand_model_from_dict(hand_model_dict)
    return _UMETRACK_HAND_MODEL


def _mirror_hand_model(profile: HandModel) -> HandModel:
    mirrored_joint_rotation_axes = profile.joint_rotation_axes.clone()
    mirrored_joint_rest_positions = profile.joint_rest_positions.clone()
    mirrored_mesh_vertices = profile.mesh_vertices.clone() if profile.mesh_vertices is not None else None
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
    tris_np = triangles.cpu().numpy().copy()
    if flip:
        tris_np = tris_np[:, [0, 2, 1]]  # flip winding to fix inside-out normals
    return vertices.cpu().numpy(), tris_np


def _get_fk_hand_model():
    global _FK_HAND_MODEL
    if _FK_HAND_MODEL is None:
        _FK_HAND_MODEL = load_default_hand_model()
    return _FK_HAND_MODEL


def _get_mano_layer(device="cpu"):
    global _MANO_LAYER
    if _MANO_LAYER is None:
        _MANO_LAYER = smplx.MANO(
            model_path=MANO_MODEL_PATH,
            is_rhand=True,
            flat_hand_mean=False,
            use_pca=False,
            num_pca_comps=45,
        ).to(device)
    return _MANO_LAYER


# ── Collate ──────────────────────────────────────────────────────────────────

class _DatasetWithMeta(torch.utils.data.Dataset):
    """Wrapper that adds ep_idx, center_idx, and abs_idx to each sample."""

    def __init__(self, base: EgoEmgMemmapDataset, indices: Optional[list[int]] = None):
        self.base = base
        self.indices = indices  # if using a subset mapping

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.base)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx] if self.indices is not None else idx
        sample = self.base[real_idx]
        ep_idx, center_idx = self.base._resolve_index_to_center(real_idx)
        sample["_ep_idx"] = ep_idx
        sample["_center_idx"] = center_idx
        sample["_abs_idx"] = real_idx
        return sample


def _collate_fusion(batch: list[dict]) -> dict:
    from torch.utils.data._utils.collate import default_collate

    emg_batch = []
    for sample in batch:
        emg = sample["emg"]
        if isinstance(emg, np.ndarray):
            emg = torch.as_tensor(emg, dtype=torch.float32)
        ja = sample.get("joint_angles")
        if isinstance(ja, np.ndarray):
            ja = torch.as_tensor(ja, dtype=torch.float32)
        mask = sample.get("label_valid_mask")
        if isinstance(mask, np.ndarray):
            mask = torch.as_tensor(mask, dtype=torch.bool)
        vf = sample.get("vision_features")
        if isinstance(vf, np.ndarray):
            vf = torch.as_tensor(vf, dtype=torch.float32)
        vf_mask = sample.get("vision_valid_mask")
        if isinstance(vf_mask, np.ndarray):
            vf_mask = torch.as_tensor(vf_mask, dtype=torch.bool)
        vi = sample.get("vision_img")
        if isinstance(vi, np.ndarray):
            vi = torch.as_tensor(vi, dtype=torch.float32)

        item = {"emg": emg, "joint_angles": ja, "label_valid_mask": mask}
        if vf is not None:
            item["vision_features"] = vf
        if vf_mask is not None:
            item["vision_valid_mask"] = vf_mask
        if vi is not None:
            item["vision_img"] = vi
        # Pass through metadata
        for key in ["_ep_idx", "_center_idx", "_abs_idx", "target_hand",
                     "vision_frame_indices"]:
            if key in sample:
                item[key] = sample[key]
        emg_batch.append(item)

    return default_collate(emg_batch)


# ── Model loading ────────────────────────────────────────────────────────────

def load_module(ckpt_path: str, config) -> EmgPredictionModule:
    module = EmgPredictionModule(
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.get("lr_scheduler"),
        loss_weights=config.get("loss_weights", {}),
        datamodule=config.get("datamodule"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    module.eval()
    return module


# ── Per-sample evaluation ────────────────────────────────────────────────────

def evaluate_per_sample(
    module: EmgPredictionModule,
    dataloader: DataLoader,
    device: torch.device,
    inner_model,
    emg_model: Emg2PoseFormer | None = None,
    vis_model: VisionViTPose | None = None,
) -> list[dict]:
    """Return per-sample dicts with vision_mae, fusion_mae, delta, and metadata."""
    module.to(device)
    results: list[dict] = []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        with torch.no_grad():
            # ── Fusion model forward ──────────────────────────────────────
            out = module.forward(batch_gpu)
            if isinstance(out, tuple):
                preds, targets, mask = out
            else:
                preds = out
                targets = batch_gpu["joint_angles"]
                mask = batch_gpu["label_valid_mask"]

            preds_fusion = preds.squeeze(-1)  # (B, 22)
            if targets.ndim == 3:
                targets = targets.squeeze(-1)

            # ── Standalone vision model ───────────────────────────────────
            if vis_model is not None and "vision_img" in batch_gpu:
                vis_out = vis_model(batch_gpu)
                if isinstance(vis_out, tuple):
                    y_v_all = vis_out[0].squeeze(-1)  # (B, 22, 1) → (B, 22)
                else:
                    y_v_all = vis_out.squeeze(-1)
            else:
                y_v_all = torch.zeros(preds_fusion.shape[0], preds_fusion.shape[1],
                                      device=device)

            # ── Standalone EMG model ─────────────────────────────────────
            if emg_model is not None:
                emg_preds_full = emg_model(batch_gpu)  # (B, 22, T)
                if isinstance(emg_preds_full, tuple):
                    emg_preds_full = emg_preds_full[0]
                t_center = emg_preds_full.shape[-1] // 2
                y_emg_all = emg_preds_full[:, :, t_center]  # (B, 22)
            else:
                y_emg_all = torch.zeros(preds_fusion.shape[0], preds_fusion.shape[1],
                                        device=device)

        vision_img = batch.get("vision_img")
        ep_idx_list = batch.get("_ep_idx")
        center_idx_list = batch.get("_center_idx")
        abs_idx_list = batch.get("_abs_idx")
        vf_indices = batch.get("vision_frame_indices")
        target_hands = batch.get("target_hand")

        for i in range(preds_fusion.shape[0]):
            per_joint_v = (y_v_all[i] - targets[i]).abs()
            per_joint_f = (preds_fusion[i] - targets[i]).abs()
            per_joint_e = (y_emg_all[i] - targets[i]).abs()
            mae_vision = float(per_joint_v.mean())
            mae_fusion = float(per_joint_f.mean())
            mae_emg = float(per_joint_e.mean())

            result = {
                "mae_vision": mae_vision,
                "mae_fusion": mae_fusion,
                "mae_emg": mae_emg,
                "delta_mae": mae_vision - mae_fusion,
                "per_joint_vision": per_joint_v.cpu().numpy(),
                "per_joint_fusion": per_joint_f.cpu().numpy(),
                "per_joint_emg": per_joint_e.cpu().numpy(),
                "y_v": y_v_all[i].cpu().numpy(),
                "y_emg": y_emg_all[i].cpu().numpy(),
                "pred_fusion": preds_fusion[i].cpu().numpy(),
                "target": targets[i].cpu().numpy(),
            }
            if vision_img is not None:
                result["vision_img"] = vision_img[i].cpu().numpy()
            if ep_idx_list is not None:
                result["ep_idx"] = int(ep_idx_list[i]) if isinstance(ep_idx_list, torch.Tensor) else ep_idx_list[i]
            if center_idx_list is not None:
                result["center_idx"] = int(center_idx_list[i]) if isinstance(center_idx_list, torch.Tensor) else center_idx_list[i]
            if abs_idx_list is not None:
                result["abs_idx"] = int(abs_idx_list[i]) if isinstance(abs_idx_list, torch.Tensor) else abs_idx_list[i]
            if vf_indices is not None:
                vi_val = vf_indices[i]
                result["video_frame_idx"] = int(vi_val) if isinstance(vi_val, (torch.Tensor, np.ndarray, np.integer)) else vi_val
            if target_hands is not None:
                th = target_hands[i]
                result["target_hand"] = th if isinstance(th, str) else str(th)
            results.append(result)

    return results


# ── Projection helpers (matching verify_training_with_emg.py) ─────────────────

def build_intrinsics_for_frame(frame_bgr, K_raw, dist_raw, calib_w, calib_h):
    video_h, video_w = frame_bgr.shape[:2]
    K_use, dist_use, info, _ = build_intrinsics_and_frame_mapper(
        K_raw, dist_raw, calib_w, calib_h, video_w, video_h,
        mode="gopro_8x7_crop_upsample",
        first_frame=frame_bgr,
    )
    return K_use, dist_use, info, video_w, video_h


def project_points(pts_world, T_W_C, K, dist, info):
    pts_proc, depth_valid = _project_world_points(pts_world, T_W_C, K, dist)
    pts_raw = _map_processed_points_to_raw(pts_proc, info)
    return pts_raw, depth_valid


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
    return (max(0, cx - sz), max(0, cy - sz),
            min(img_w, cx + sz), min(img_h, cy + sz))


def crop_hand_region(frame_bgr, bbox, target_size=256):
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_skeleton_2d(img_bgr, pts, valid, color, label=None, linewidth=2,
                     nodesize=3):
    img = img_bgr.copy()
    valid_b = np.asarray(valid, dtype=bool)
    for i0, i1 in SKELETON_EDGES:
        if i0 >= len(pts) or i1 >= len(pts):
            continue
        if valid_b[i0] and valid_b[i1]:
            p0 = tuple(np.round(pts[i0]).astype(np.int32))
            p1 = tuple(np.round(pts[i1]).astype(np.int32))
            cv2.line(img, p0, p1, color, linewidth, lineType=cv2.LINE_AA)
    for i, (p, v) in enumerate(zip(pts, valid_b)):
        if not v:
            continue
        cv2.circle(img, tuple(np.round(p).astype(np.int32)),
                   nodesize, color, -1, lineType=cv2.LINE_AA)
    if label and valid_b.any():
        cy, cx = pts[valid_b].mean(axis=0).astype(np.int32)
        cv2.putText(img, label, (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
    return img


def draw_wireframe(image_bgr, points_xy, valid, faces, color_bgr, line_width=1):
    out = image_bgr.copy()
    valid_b = np.asarray(valid, dtype=bool)
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid_b[i0] and valid_b[i1] and valid_b[i2]):
            continue
        p0 = tuple(np.round(points_xy[i0]).astype(np.int32))
        p1 = tuple(np.round(points_xy[i1]).astype(np.int32))
        p2 = tuple(np.round(points_xy[i2]).astype(np.int32))
        for a, b in [(p0, p1), (p1, p2), (p2, p0)]:
            cv2.line(out, a, b, color_bgr, line_width, lineType=cv2.LINE_AA)
    return out


# ── Shaded mesh overlay (per-triangle Lambertian with correct projection) ──────


def render_mesh_shaded_overlay(frame_bgr, verts_world, faces, T_W_C, K_use,
                                dist_use, info, video_w, video_h,
                                color=(0.75, 0.75, 0.75), alpha=0.60):
    """Render a shaded mesh overlay using per-triangle flat shading with the
    correct camera projection (including distortion).

    Uses cv2.projectPoints internally (via _project_world_points) which correctly
    handles points behind the camera and lens distortion.  Triangles are
    depth-sorted back-to-front and shaded with Lambertian reflectance.
    """
    if verts_world is None or len(verts_world) == 0:
        return frame_bgr

    # Project all vertices using the full pipeline (handles distortion + Z clipping)
    verts_proc, depth_valid = _project_world_points(
        verts_world, T_W_C, K_use, dist_use)
    verts_2d = _map_processed_points_to_raw(verts_proc, info)

    in_bounds = (
        depth_valid
        & (verts_2d[:, 0] >= -2000) & (verts_2d[:, 0] < video_w + 2000)
        & (verts_2d[:, 1] >= -2000) & (verts_2d[:, 1] < video_h + 2000)
        & np.isfinite(verts_2d).all(axis=1)
    )

    if in_bounds.sum() < 3:
        return frame_bgr

    # Camera-space vertices for shading — must match convention used by
    # _project_world_points (inverted T_W_C rotation, so Z > 0 is forward).
    T_C_W = np.linalg.inv(T_W_C)
    R_C_W = T_C_W[:3, :3].astype(np.float64)
    t_C_W = T_C_W[:3, 3].astype(np.float64)
    verts_cam = (R_C_W @ verts_world.T).T + t_C_W

    # Compute vertex normals (average of adjacent face normals)
    n_verts = len(verts_world)
    vertex_normals = np.zeros((n_verts, 3), dtype=np.float64)
    face_centers_cam = np.zeros((len(faces), 3), dtype=np.float64)
    face_normals = np.zeros((len(faces), 3), dtype=np.float64)
    face_visible = np.ones(len(faces), dtype=bool)

    for fi, tri in enumerate(faces):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (in_bounds[i0] and in_bounds[i1] and in_bounds[i2]):
            face_visible[fi] = False
            continue
        v0, v1, v2 = verts_cam[i0], verts_cam[i1], verts_cam[i2]
        normal = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(normal)
        if nlen < 1e-12:
            face_visible[fi] = False
            continue
        normal /= nlen
        face_normals[fi] = normal
        face_centers_cam[fi] = (v0 + v1 + v2) / 3.0
        vertex_normals[i0] += normal
        vertex_normals[i1] += normal
        vertex_normals[i2] += normal

    # Normalize vertex normals
    for i in range(n_verts):
        nlen = np.linalg.norm(vertex_normals[i])
        if nlen > 1e-12:
            vertex_normals[i] /= nlen

    # Compute per-face shade from vertex normals (Gouraud-style via per-vertex average)
    face_shades = np.zeros(len(faces), dtype=np.float64)
    for fi, tri in enumerate(faces):
        if not face_visible[fi]:
            continue
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        center = face_centers_cam[fi]
        view_dir = -center / (np.linalg.norm(center) + 1e-12)
        # Average the three vertex normals for the face
        avg_normal = vertex_normals[i0] + vertex_normals[i1] + vertex_normals[i2]
        avg_normal /= np.linalg.norm(avg_normal) + 1e-12
        n_dot_v = np.dot(avg_normal, view_dir)
        # Lambertian: ambient (0.35) + directional. max(0, n_dot_v) so
        # backfaces get only ambient — this gives proper 3D depth cues.
        face_shades[fi] = 0.35 + 0.65 * max(0.0, n_dot_v)

    # Bucket triangles by shade level (10 buckets for ~10 fillPoly calls)
    n_buckets = 10
    bucket_tris: dict[int, list] = {i: [] for i in range(n_buckets)}
    for fi, tri in enumerate(faces):
        if not face_visible[fi]:
            continue
        bucket_idx = min(int(face_shades[fi] * n_buckets), n_buckets - 1)
        pts = np.array([
            np.round(verts_2d[int(tri[0])]).astype(np.int32),
            np.round(verts_2d[int(tri[1])]).astype(np.int32),
            np.round(verts_2d[int(tri[2])]).astype(np.int32),
        ])
        bucket_tris[bucket_idx].append(pts)

    # Sort faces by depth (back-to-front) within each bucket
    # Compute per-face centroid Z for sorting
    face_z = np.zeros(len(faces), dtype=np.float64)
    for fi, tri in enumerate(faces):
        if face_visible[fi]:
            face_z[fi] = face_centers_cam[fi][2]  # Z in camera space

    # Re-collect sorted by Z
    sorted_indices = np.argsort(face_z[face_visible])  # farthest first
    visible_indices = np.where(face_visible)[0]
    sorted_face_indices = visible_indices[sorted_indices]

    # Compute per-face shade bucket
    overlay = np.zeros((video_h, video_w, 3), dtype=np.float32)
    mask_acc = np.zeros((video_h, video_w), dtype=np.float32)

    base_color_255 = np.array([c * 255.0 for c in color], dtype=np.float32)

    for fi in sorted_face_indices:
        shade = face_shades[fi]
        tri = faces[fi]
        pts = np.array([
            np.round(verts_2d[int(tri[0])]).astype(np.int32),
            np.round(verts_2d[int(tri[1])]).astype(np.int32),
            np.round(verts_2d[int(tri[2])]).astype(np.int32),
        ])
        # Check if triangle is within image
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= video_w:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= video_h:
            continue

        shaded = base_color_255 * shade
        cv2.fillPoly(overlay, [pts], shaded.tolist())
        cv2.fillPoly(mask_acc, [pts], alpha)

    if mask_acc.max() < 1e-6:
        return frame_bgr

    result = frame_bgr.astype(np.float32).copy()
    result = result * (1.0 - mask_acc[..., None]) + overlay * mask_acc[..., None]
    return result.clip(0, 255).astype(np.uint8)


# ── FK keypoints in world space ──────────────────────────────────────────────

def fk_keypoints_world(joint_angles_20: np.ndarray, flip: bool = False):
    """Compute 21 FK keypoints in local FK space, with optional X-flip."""
    from emg2pose.kinematics import forward_kinematics
    hand_model = _get_fk_hand_model()
    # forward_kinematics expects (B, J, T) — add time dim
    ja_t = torch.from_numpy(np.asarray(joint_angles_20, dtype=np.float32))
    ja_t = ja_t.unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
    kp = forward_kinematics(ja_t, hand_model)  # (1, T, 21, 3)
    kp = kp[0, 0].cpu().numpy()  # (21, 3)
    if flip:
        kp = kp * MIRROR_X_3
    return kp


# ── GLB export ──────────────────────────────────────────────────────────────

def save_mesh_glb(vertices_world, faces, path, color=(100, 180, 100)):
    if vertices_world is None or not np.isfinite(vertices_world).all():
        mesh = trimesh.creation.icosphere(radius=0.001)
        mesh.export(str(path))
        return
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)
    mesh.visual.vertex_colors = (*color, 255)
    mesh.export(str(path))


# ── Rich visualization for one sample ────────────────────────────────────────

def visualize_sample(
    result: dict,
    output_dir: Path,
    rank: int,
    memmap_dir: Path,
    video_root: Path,
    dataset: EgoEmgMemmapDataset,
    device: torch.device,
    calib: dict | None = None,
) -> None:
    """Generate full visualizations for a single sample."""
    sample_dir = output_dir / f"sample_{rank+1:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    hand = result.get("target_hand", "right")
    flip = (hand == "left")
    ep_idx = result.get("ep_idx")
    center_idx = result.get("center_idx")
    vf_idx = result.get("video_frame_idx")

    # ── Read memmap data for this sample ──────────────────────────────────
    ep_id = dataset._episode_id[ep_idx]
    if isinstance(ep_id, (bytes, np.bytes_)):
        ep_id = ep_id.decode("utf-8").rstrip("\x00")
    ep_id = str(ep_id)

    # Video path
    raw_video_rel = dataset._episode_webcam_video_path[ep_idx]
    if isinstance(raw_video_rel, (bytes, np.bytes_)):
        raw_video_rel = raw_video_rel.decode("utf-8").rstrip("\x00")
    video_path = str(video_root / str(raw_video_rel).replace(".mp4", "_allintra.mp4"))

    # Frame index (load from memmap — may not be in dataset's modality groups)
    if vf_idx is None and center_idx is not None:
        fi_mm = _load_mm(memmap_dir, "image_webcam_frame_index")
        vf_idx = int(fi_mm[center_idx])

    # ── Read video frame ──────────────────────────────────────────────────
    try:
        from decord import VideoReader, cpu as decord_cpu
        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        vf_idx_clamped = max(0, min(int(vf_idx), len(vr) - 1))
        frame_rgb = vr[vf_idx_clamped].asnumpy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        print(f"  [WARN] Cannot read video frame for {ep_id}:{vf_idx}, using blank")
        frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)

    orig_clean = frame_bgr.copy()

    # ── Calibration & intrinsics ──────────────────────────────────────────
    if calib is None:
        calib_path = (_DATA_ROOT / "reprojection_assets" /
                      "GX010023_standard_calibration.json")
        with open(calib_path) as f:
            calib = json.load(f)
    K_raw = np.asarray(calib["camera_matrix"], dtype=np.float64)
    dist_raw = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    calib_w = int(calib["image_width"])
    calib_h = int(calib["image_height"])

    K_use, dist_use, info, video_w, video_h = build_intrinsics_for_frame(
        frame_bgr, K_raw, dist_raw, calib_w, calib_h)

    # ── Camera transform (direct memmap load) ─────────────────────────────
    cam_transform_mm = _load_mm(memmap_dir, "mocap_webcam_transform")
    t12 = np.asarray(cam_transform_mm[center_idx], dtype=np.float64)
    T_W_C = np.eye(4, dtype=np.float64)
    T_W_C[:3, :3] = t12[:9].reshape(3, 3)
    T_W_C[:3, 3] = t12[9:12]

    # ── Markers (direct memmap load) ──────────────────────────────────────
    kp_mm = _load_mm(memmap_dir, f"mocap_{hand}_keypoints")
    valid_mm = _load_mm(memmap_dir, f"mocap_{hand}_valid")
    kp_world = np.asarray(kp_mm[center_idx], dtype=np.float64)
    valid_kp = np.asarray(valid_mm[center_idx], dtype=bool)

    pts_marker, depth_valid = project_points(kp_world, T_W_C, K_use, dist_use, info)
    valid_marker = (
        depth_valid
        & (pts_marker[:, 0] >= 0) & (pts_marker[:, 0] < video_w)
        & (pts_marker[:, 1] >= 0) & (pts_marker[:, 1] < video_h)
        & valid_kp
        & np.isfinite(kp_world).all(axis=1)
    )

    # ── MANO world transform & mesh (direct memmap load) ──────────────────
    mano_world_mm = _load_mm(memmap_dir, f"mocap_mano_{hand}_world_transform")
    t12_world = np.asarray(mano_world_mm[center_idx], dtype=np.float64)
    R_world = t12_world[:9].reshape(3, 3)
    t_world = t12_world[9:12]

    # MANO pose (direct memmap load)
    mano_pose_mm = _load_mm(memmap_dir, f"generated_mano_{hand}_pose")
    mano_pose = np.asarray(mano_pose_mm[center_idx], dtype=np.float32)

    # MANO beta (episode-level field — loaded from metadata, indexed by ep_idx)
    try:
        md = np.load(memmap_dir / "metadata.npz", allow_pickle=False)
        beta_arr = md[f"generated_mano_{hand}_beta"][ep_idx]
        beta = np.asarray(beta_arr, dtype=np.float32)
    except (KeyError, IndexError):
        beta = np.zeros(10, dtype=np.float32)

    mano_verts_world = None
    mano_faces_out = None
    mano_layer = _get_mano_layer(str(device))
    mano_faces = mano_layer.faces.copy()

    if np.isfinite(mano_pose).all() and np.abs(mano_pose).sum() > 0:
        try:
            global_orient = torch.zeros(1, 3, dtype=torch.float32, device=device)
            hand_pose_aa = mano_pose[3:48].astype(np.float32)
            hp_t = torch.tensor(hand_pose_aa, dtype=torch.float32, device=device).unsqueeze(0)
            betas_t = torch.tensor(beta, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                out_data = mano_layer(global_orient=global_orient, hand_pose=hp_t, betas=betas_t)
            verts_local = out_data.vertices[0].cpu().numpy()
            mano_faces_out = mano_faces.copy()
            if flip:
                verts_local = verts_local * MIRROR_X_3
                mano_faces_out = mano_faces_out[:, [0, 2, 1]]
            mano_verts_world = (R_world @ verts_local.T).T + t_world
        except Exception as e:
            tqdm.write(f"  MANO decode failed: {e}")

    # MANO projection
    mano_pts_raw = None
    mano_valid = None
    if mano_verts_world is not None:
        verts_proc, depth_m = _project_world_points(mano_verts_world, T_W_C, K_use, dist_use)
        mano_pts_raw = _map_processed_points_to_raw(verts_proc, info)
        mano_valid = (
            depth_m
            & (mano_pts_raw[:, 0] >= 0) & (mano_pts_raw[:, 0] < video_w)
            & (mano_pts_raw[:, 1] >= 0) & (mano_pts_raw[:, 1] < video_h)
        )

    # ── FK meshes from predictions & GT ───────────────────────────────────
    # GT FK
    gt_ja = result["target"][:20]  # first 20 angles
    gt_fk_local, gt_fk_faces = skin_fk_mesh(joint_angles=gt_ja, flip=flip)
    gt_fk_local = gt_fk_local.copy()
    fk_span = np.median(gt_fk_local.max(axis=0) - gt_fk_local.min(axis=0))
    if fk_span > 1e-6:
        gt_fk_local = gt_fk_local * (0.09 / fk_span)
    gt_fk_world = (R_world @ gt_fk_local.T).T + t_world

    # Vision FK
    vis_ja = result["y_v"][:20]
    vis_fk_local, vis_fk_faces = skin_fk_mesh(joint_angles=vis_ja, flip=flip)
    vis_fk_local = vis_fk_local.copy()
    fk_span_v = np.median(vis_fk_local.max(axis=0) - vis_fk_local.min(axis=0))
    if fk_span_v > 1e-6:
        vis_fk_local = vis_fk_local * (0.09 / fk_span_v)
    vis_fk_world = (R_world @ vis_fk_local.T).T + t_world

    # Fusion FK
    fus_ja = result["pred_fusion"][:20]
    fus_fk_local, fus_fk_faces = skin_fk_mesh(joint_angles=fus_ja, flip=flip)
    fus_fk_local = fus_fk_local.copy()
    fk_span_f = np.median(fus_fk_local.max(axis=0) - fus_fk_local.min(axis=0))
    if fk_span_f > 1e-6:
        fus_fk_local = fus_fk_local * (0.09 / fk_span_f)
    fus_fk_world = (R_world @ fus_fk_local.T).T + t_world

    # EMG-only FK
    emg_ja = result["y_emg"][:20]
    emg_fk_local, emg_fk_faces = skin_fk_mesh(joint_angles=emg_ja, flip=flip)
    emg_fk_local = emg_fk_local.copy()
    fk_span_e = np.median(emg_fk_local.max(axis=0) - emg_fk_local.min(axis=0))
    if fk_span_e > 1e-6:
        emg_fk_local = emg_fk_local * (0.09 / fk_span_e)
    emg_fk_world = (R_world @ emg_fk_local.T).T + t_world

    # ── Project FK meshes (wireframe via keypoints) ───────────────────────
    def project_fk_mesh(verts_world, T_W_C, K_use, dist_use, info):
        v_proc, d_valid = _project_world_points(verts_world, T_W_C, K_use, dist_use)
        v_raw = _map_processed_points_to_raw(v_proc, info)
        v_ok = (d_valid & (v_raw[:, 0] >= 0) & (v_raw[:, 0] < video_w)
                & (v_raw[:, 1] >= 0) & (v_raw[:, 1] < video_h))
        return v_raw, v_ok

    fus_pts_raw, fus_valid_proj = project_fk_mesh(fus_fk_world, T_W_C, K_use, dist_use, info)
    vis_pts_raw, vis_valid_proj = project_fk_mesh(vis_fk_world, T_W_C, K_use, dist_use, info)
    gt_pts_raw, gt_valid_proj = project_fk_mesh(gt_fk_world, T_W_C, K_use, dist_use, info)
    emg_pts_raw, emg_valid_proj = project_fk_mesh(emg_fk_world, T_W_C, K_use, dist_use, info)

    # ── FK keypoints projection (21 keypoints for skeleton overlay) ───────
    # fk_keypoints_world returns mm-scale in local space. The FK mesh is
    # normalized to 0.09m span before world transform. Scale keypoints the
    # same way so they land at the correct world position.
    gt_kp_local = fk_keypoints_world(gt_ja, flip=flip)
    gt_kp_local = gt_kp_local * (0.09 / fk_span) if fk_span > 1e-6 else gt_kp_local / 1000.0
    vis_kp_local = fk_keypoints_world(vis_ja, flip=flip)
    vis_kp_local = vis_kp_local * (0.09 / fk_span_v) if fk_span_v > 1e-6 else vis_kp_local / 1000.0
    fus_kp_local = fk_keypoints_world(fus_ja, flip=flip)
    fus_kp_local = fus_kp_local * (0.09 / fk_span_f) if fk_span_f > 1e-6 else fus_kp_local / 1000.0
    emg_kp_local = fk_keypoints_world(emg_ja, flip=flip)
    emg_kp_local = emg_kp_local * (0.09 / fk_span_e) if fk_span_e > 1e-6 else emg_kp_local / 1000.0

    def project_kp(kp_local):
        kp_w = (R_world @ kp_local.T).T + t_world
        return project_points(kp_w, T_W_C, K_use, dist_use, info)

    gt_kp_2d, gt_kp_valid = project_kp(gt_kp_local)
    vis_kp_2d, vis_kp_valid = project_kp(vis_kp_local)
    fus_kp_2d, fus_kp_valid = project_kp(fus_kp_local)
    emg_kp_2d, emg_kp_valid = project_kp(emg_kp_local)

    # ── Bbox from markers ─────────────────────────────────────────────────
    bbox = get_points_bbox(pts_marker, valid_marker, video_w, video_h, margin=30)
    crop_clean = crop_hand_region(orig_clean, bbox, target_size=256)

    # ── Bbox from MANO ────────────────────────────────────────────────────
    mano_bbox = None
    if mano_pts_raw is not None and mano_valid is not None:
        mano_bbox = get_points_bbox(mano_pts_raw, mano_valid, video_w, video_h, margin=20)

    # ── Draw on original ──────────────────────────────────────────────────
    # Markers
    markers_orig = orig_clean.copy()
    if valid_marker.sum() > 0:
        markers_orig = draw_skeleton_2d(markers_orig, pts_marker, valid_marker,
                                        color=(0, 255, 255), label=hand[0].upper())
    cv2.imwrite(str(sample_dir / "markers_proj_original.png"), markers_orig)

    # MANO mesh shaded overlay (grey-white)
    mano_orig = orig_clean.copy()
    if mano_verts_world is not None and mano_faces_out is not None:
        mano_orig = render_mesh_shaded_overlay(
            mano_orig, mano_verts_world, mano_faces_out, T_W_C, K_use,
            dist_use, info, video_w, video_h, color=(0.82, 0.80, 0.78), alpha=0.55)
    cv2.imwrite(str(sample_dir / "mano_proj_original.png"), mano_orig)

    # MANO bbox
    bbox_orig = orig_clean.copy()
    if mano_bbox is not None:
        cv2.rectangle(bbox_orig, (mano_bbox[0], mano_bbox[1]),
                      (mano_bbox[2], mano_bbox[3]), (0, 255, 0), 2)
    cv2.imwrite(str(sample_dir / "mano_bbox_original.png"), bbox_orig)

    # GT FK mesh shaded overlay + skeleton
    gt_skel = orig_clean.copy()
    gt_skel = render_mesh_shaded_overlay(
        gt_skel, gt_fk_world, gt_fk_faces, T_W_C, K_use,
        dist_use, info, video_w, video_h, color=(0.60, 0.78, 0.60), alpha=0.55)
    if gt_kp_valid.sum() > 0:
        gt_skel = draw_skeleton_2d(gt_skel, gt_kp_2d, gt_kp_valid,
                                   color=(0, 200, 0), label="GT")
    cv2.imwrite(str(sample_dir / "gt_skeleton_original.png"), gt_skel)

    # Vision FK mesh shaded overlay + skeleton
    vis_skel = orig_clean.copy()
    vis_skel = render_mesh_shaded_overlay(
        vis_skel, vis_fk_world, vis_fk_faces, T_W_C, K_use,
        dist_use, info, video_w, video_h, color=(0.82, 0.45, 0.45), alpha=0.55)
    if vis_kp_valid.sum() > 0:
        vis_skel = draw_skeleton_2d(vis_skel, vis_kp_2d, vis_kp_valid,
                                    color=(0, 0, 255), label="Vis")
    cv2.imwrite(str(sample_dir / "vision_skeleton_original.png"), vis_skel)

    # Fusion FK mesh shaded overlay + skeleton
    fus_skel = orig_clean.copy()
    fus_skel = render_mesh_shaded_overlay(
        fus_skel, fus_fk_world, fus_fk_faces, T_W_C, K_use,
        dist_use, info, video_w, video_h, color=(0.55, 0.78, 0.60), alpha=0.55)
    if fus_kp_valid.sum() > 0:
        fus_skel = draw_skeleton_2d(fus_skel, fus_kp_2d, fus_kp_valid,
                                    color=(0, 255, 0), label="Fus")
    cv2.imwrite(str(sample_dir / "fusion_skeleton_original.png"), fus_skel)

    # EMG-only FK mesh shaded overlay + skeleton
    emg_skel = orig_clean.copy()
    emg_skel = render_mesh_shaded_overlay(
        emg_skel, emg_fk_world, emg_fk_faces, T_W_C, K_use,
        dist_use, info, video_w, video_h, color=(0.45, 0.55, 0.82), alpha=0.55)
    if emg_kp_valid.sum() > 0:
        emg_skel = draw_skeleton_2d(emg_skel, emg_kp_2d, emg_kp_valid,
                                    color=(255, 165, 0), label="EMG")
    cv2.imwrite(str(sample_dir / "emg_skeleton_original.png"), emg_skel)

    # ── Side-by-side comparison (original) ────────────────────────────────
    panel_w = video_w // 4
    panel_h = video_h // 4
    comp_orig = np.hstack([
        cv2.resize(vis_skel, (panel_w, panel_h)),
        cv2.resize(emg_skel, (panel_w, panel_h)),
        cv2.resize(gt_skel, (panel_w, panel_h)),
        cv2.resize(fus_skel, (panel_w, panel_h)),
    ])
    for j, label in enumerate(["Vision-only", "EMG-only", "Ground Truth", "Fusion"]):
        x = j * panel_w + 10
        cv2.putText(comp_orig, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(sample_dir / "comparison_original.png"), comp_orig)

    # ── Draw on crop ──────────────────────────────────────────────────────
    def draw_on_crop(crop_img, bbox, pts, valid, color, label):
        if crop_img is None or bbox is None:
            return None
        x0, y0, _, _ = bbox
        pts_c = pts.copy()
        pts_c[:, 0] -= x0
        pts_c[:, 1] -= y0
        scale = 256.0 / (bbox[2] - bbox[0])
        pts_c[:, 0] *= scale
        pts_c[:, 1] *= scale
        in_c = (pts_c[:, 0] >= 0) & (pts_c[:, 0] < 256) & (pts_c[:, 1] >= 0) & (pts_c[:, 1] < 256)
        v_c = valid & in_c
        return draw_skeleton_2d(crop_img.copy(), pts_c, v_c, color, label)

    if crop_clean is not None:
        cv2.imwrite(str(sample_dir / "crop.png"), crop_clean)
        # Markers on crop
        mc = draw_on_crop(crop_clean, bbox, pts_marker, valid_marker, (0, 255, 255), hand[0].upper())
        if mc is not None:
            cv2.imwrite(str(sample_dir / "markers_proj_crop.png"), mc)
        # GT on crop
        gc = draw_on_crop(crop_clean, bbox, gt_kp_2d, gt_kp_valid, (0, 255, 0), "GT")
        if gc is not None:
            cv2.imwrite(str(sample_dir / "gt_skeleton_crop.png"), gc)
        # Vision on crop
        vc = draw_on_crop(crop_clean, bbox, vis_kp_2d, vis_kp_valid, (0, 0, 255), "Vis")
        if vc is not None:
            cv2.imwrite(str(sample_dir / "vision_skeleton_crop.png"), vc)
        # Fusion on crop
        fc = draw_on_crop(crop_clean, bbox, fus_kp_2d, fus_kp_valid, (0, 255, 0), "Fus")
        if fc is not None:
            cv2.imwrite(str(sample_dir / "fusion_skeleton_crop.png"), fc)
        # EMG on crop
        ec = draw_on_crop(crop_clean, bbox, emg_kp_2d, emg_kp_valid, (255, 165, 0), "EMG")
        if ec is not None:
            cv2.imwrite(str(sample_dir / "emg_skeleton_crop.png"), ec)
        # MANO wireframe on crop (keep lightweight 2D for crops)
        if mano_pts_raw is not None and mano_valid is not None:
            mcrop_img = crop_clean.copy()
            x0, y0, _, _ = bbox
            m_c = mano_pts_raw.copy()
            m_c[:, 0] -= x0; m_c[:, 1] -= y0
            scale_m = 256.0 / (bbox[2] - bbox[0])
            m_c[:, 0] *= scale_m; m_c[:, 1] *= scale_m
            in_m = (m_c[:, 0] >= 0) & (m_c[:, 0] < 256) & (m_c[:, 1] >= 0) & (m_c[:, 1] < 256)
            m_vc = mano_valid & in_m
            if m_vc.sum() > 0 and mano_faces_out is not None:
                # Draw triangle edges for crop overlay
                valid_b = np.asarray(m_vc, dtype=bool)
                for tri in mano_faces_out:
                    i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
                    if not (valid_b[i0] and valid_b[i1] and valid_b[i2]):
                        continue
                    triangle = np.array([
                        np.round(m_c[i0]).astype(np.int32),
                        np.round(m_c[i1]).astype(np.int32),
                        np.round(m_c[i2]).astype(np.int32),
                    ])
                    cv2.polylines(mcrop_img, [triangle], True, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imwrite(str(sample_dir / "mano_proj_crop.png"), mcrop_img)

    # ── Original frame ────────────────────────────────────────────────────
    cv2.imwrite(str(sample_dir / "original.png"), orig_clean)

    # ── GLB exports ───────────────────────────────────────────────────────
    # GT FK mesh — neutral grey reference
    save_mesh_glb(gt_fk_world, gt_fk_faces, sample_dir / "gt_from_angles.glb",
                  color=(190, 190, 190))
    # Vision FK mesh — warm orange-red (baseline, "weaker")
    save_mesh_glb(vis_fk_world, vis_fk_faces, sample_dir / "pred_vision.glb",
                  color=(213, 94, 0))
    # Fusion FK mesh — teal-green (improvement, "better")
    save_mesh_glb(fus_fk_world, fus_fk_faces, sample_dir / "pred_fusion.glb",
                  color=(0, 158, 115))
    # EMG-only FK mesh — blue (other modality)
    save_mesh_glb(emg_fk_world, emg_fk_faces, sample_dir / "pred_emg.glb",
                  color=(0, 114, 178))
    # MANO GT
    if mano_verts_world is not None and mano_faces_out is not None:
        save_mesh_glb(mano_verts_world, mano_faces_out, sample_dir / "mano_gt.glb",
                      color=(160, 160, 160))

    # ── Per-joint error comparison chart ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1, 1.5]})

    # Left: hand crop (from dataset's normalized image)
    ax_img = axes[0]
    if "vision_img" in result and result["vision_img"] is not None:
        img = result["vision_img"].transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        ax_img.imshow(img)
    ax_img.set_title(f"Hand Crop ({hand})", fontsize=10)
    ax_img.axis("off")

    # Right: per-joint error
    ax_bar = axes[1]
    n_joints = len(JOINT_NAMES)
    x = np.arange(n_joints)
    width = 0.25
    v_err = result["per_joint_vision"] * 57.3
    e_err = result.get("per_joint_emg", np.zeros_like(v_err)) * 57.3
    f_err = result["per_joint_fusion"] * 57.3

    ax_bar.bar(x - width, v_err, width, label="Vision-only", color="#E74C3C", alpha=0.85)
    ax_bar.bar(x, e_err, width, label="EMG-only", color="#3498DB", alpha=0.85)
    ax_bar.bar(x + width, f_err, width, label="Fusion", color="#2ECC71", alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=7)
    ax_bar.set_ylabel("MAE ($^\\circ$)", fontsize=10)
    ax_bar.set_title("Per-Joint Error", fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

    for j in range(n_joints):
        delta_j = v_err[j] - f_err[j]
        if delta_j > 0.05:
            ax_bar.annotate(f"-{delta_j:.1f}$^\\circ$",
                            (x[j], max(v_err[j], e_err[j], f_err[j]) + 0.3),
                            ha="center", fontsize=5.5, color="#27AE60", fontweight="bold")

    fig.suptitle(
        f"Rank #{rank+1} | {hand} hand | "
        f"Vision={result['mae_vision']*57.3:.1f}$^\\circ$ | "
        f"EMG={result.get('mae_emg', 0)*57.3:.1f}$^\\circ$ | "
        f"Fusion={result['mae_fusion']*57.3:.1f}$^\\circ$ | "
        f"$\\Delta$={result['delta_mae']*57.3:.2f}$^\\circ$",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(sample_dir / "error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Save metadata ─────────────────────────────────────────────────────
    meta = {
        "rank": rank + 1,
        "split": result.get("split", "?"),
        "hand": hand,
        "episode": ep_id,
        "center_idx": int(center_idx) if center_idx is not None else -1,
        "video_frame": int(vf_idx) if vf_idx is not None else -1,
        "mae_vision_deg": round(result["mae_vision"] * 57.3, 2),
        "mae_emg_deg": round(result.get("mae_emg", 0) * 57.3, 2),
        "mae_fusion_deg": round(result["mae_fusion"] * 57.3, 2),
        "delta_deg": round(result["delta_mae"] * 57.3, 2),
    }
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find samples where fusion most outperforms vision-only")
    parser.add_argument("--config-name", default="experiment/fusion/vision_resnet_small_emgfusion_center")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-location", default="data/EgoEMG_v2_memmap")
    parser.add_argument("--video-root", default="data/EgoEMG_allintra")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="./fusion_wins_viz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", nargs="+", default=["user", "gesture", "both"])
    parser.add_argument("--hands", nargs="+", default=["left", "right"])
    parser.add_argument("--max-samples-per-split", type=int, default=500)
    parser.add_argument("--emg-checkpoint", default=None,
                        help="Pretrained EMG checkpoint for EMG-only head")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    exp_name = args.config_name
    for prefix in ("experiment/", "experiment\\"):
        if exp_name.startswith(prefix):
            exp_name = exp_name[len(prefix):]
            break

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        try:
            cfg = compose(config_name=args.config_name)
        except Exception:
            cfg = compose(config_name="base", overrides=[f"experiment={exp_name}"])

    print(f"Config: {args.config_name}")
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    module = load_module(args.checkpoint, cfg)
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")

    # ── Load standalone EMG model (Emg2PoseFormer) ──────────────────────────
    emg_ckpt_path = args.emg_checkpoint or cfg.get("pretrained_emg_checkpoint")
    emg_model = None
    if emg_ckpt_path and Path(emg_ckpt_path).exists():
        from hydra.utils import instantiate
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
            try:
                emg_cfg = compose(config_name="experiment/emgformer/regression_emgformer_small_aggressive_egoemg")
            except Exception:
                emg_cfg = compose(config_name="base", overrides=["experiment=emgformer/regression_emgformer_small_aggressive_egoemg"])
        emg_model = Emg2PoseFormer(
            featurizer=instantiate(emg_cfg.module.featurizer),
            decoder=instantiate(emg_cfg.module.decoder),
            head=instantiate(emg_cfg.module.head),
            out_channels=emg_cfg.module.get("out_channels", 22),
            provide_initial_pos=emg_cfg.module.get("provide_initial_pos", False),
        )
        emg_ckpt = torch.load(emg_ckpt_path, map_location="cpu", weights_only=False)
        emg_sd = emg_ckpt.get("state_dict", emg_ckpt)
        emg_sd_remapped = {k[6:] if k.startswith("model.") else k: v for k, v in emg_sd.items()}
        emg_model.load_state_dict(emg_sd_remapped, strict=True)
        emg_model.to(device)
        emg_model.eval()
        n_params = sum(p.numel() for p in emg_model.parameters())
        print(f"Loaded standalone EMG model: {n_params:,} params")

    # ── Load standalone vision model (VisionViTPose) ────────────────────────
    vis_ckpt_key = "vision_vit_checkpoint" if cfg.module.get("vision_backbone_type", "").startswith("vit") else "vision_resnet_checkpoint"
    vis_ckpt_path = cfg.get(vis_ckpt_key)
    vis_model = None
    if vis_ckpt_path and Path(vis_ckpt_path).exists():
        vis_ckpt = torch.load(vis_ckpt_path, map_location="cpu", weights_only=False)
        vis_sd = vis_ckpt.get("state_dict", vis_ckpt)
        vis_sd_remapped = {k[6:] if k.startswith("model.") else k: v for k, v in vis_sd.items()}
        vis_backbone = cfg.module.get("vision_backbone_type", "resnet18")
        if vis_backbone.startswith("vit"):
            VisModelClass = VisionViTPose
        else:
            VisModelClass = ResNetVisionPose
        vis_model = VisModelClass(
            out_channels=22,
            backbone_type=vis_backbone,
            pretrained=False,  # weights come from checkpoint
            head_hidden=512,
            head_dropout=0.1,
        )
        vis_model.load_state_dict(vis_sd_remapped, strict=True)
        vis_model.to(device)
        vis_model.eval()
        n_params = sum(p.numel() for p in vis_model.parameters())
        print(f"Loaded standalone vision model: {n_params:,} params")

    # ── Monkey-patch to extract fusion internals ────────────────────────────
    original_forward_cs = module.model._forward_center_supervised

    def patched_forward_cs(self, batch, vision_features, emg):
        emg_features = self.featurizer(emg)
        decoded = self.decoder(emg_features)
        attn_scores = self.temporal_attn(decoded.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=1)
        emg_pooled = (decoded * attn_weights.squeeze(-1).unsqueeze(1)).sum(dim=-1)
        if "vision_valid_mask" in batch:
            vision_valid = batch["vision_valid_mask"]
            if vision_valid.ndim > 1:
                vision_valid = vision_valid.any(dim=1)
            vision_features = vision_features * vision_valid[:, None].to(vision_features.dtype)
        y_v = self.head_vision(vision_features)
        vis_feat = self.vision_proj(vision_features)
        fused = torch.cat([emg_pooled, vis_feat], dim=-1).unsqueeze(-1)
        fused = self.fusion_proj(fused)
        delta = self.head(fused)
        self._last_delta = delta
        preds = y_v.unsqueeze(-1) + delta
        if "joint_angles" in batch and "label_valid_mask" in batch:
            ja = batch["joint_angles"]
            mask = batch["label_valid_mask"]
            if ja.shape[-1] == 1:
                return preds, ja, mask[..., :1] if mask.ndim >= 2 else mask.unsqueeze(-1)
            half_ctx = self.left_context // 2
            right_stop = -half_ctx if half_ctx > 0 else None
            targets_full = ja[..., half_ctx:right_stop]
            mask_full = mask[..., half_ctx:right_stop]
            center = targets_full.shape[-1] // 2
            targets = targets_full[:, :, center:center + 1]
            mask_out = mask_full[:, :, center:center + 1] if mask_full.ndim >= 3 else mask_full[..., center:center + 1]
            return preds, targets, mask_out
        return preds

    module.model._forward_center_supervised = types.MethodType(patched_forward_cs, module.model)
    inner_model = module.model

    # ── Build datasets & dataloaders ──────────────────────────────────────────
    memmap_dir = args.data_location
    emg_layout = cfg.get("egoemg_emg_layout", "emg2pose_interpolate16")
    channel_indices = cfg.get("egoemg_emg2pose_channel_indices", [10, 12, 0, 1, 2, 4, 5, 6])
    channel_interpolate = cfg.get("egoemg_channel_interpolate", False)
    norm_stats_path = cfg.datamodule.get("per_dataset_norm_stats_path")
    val_window = cfg.datamodule.get("val_test_window_length", 7790)
    val_stride = cfg.datamodule.get("val_test_stride", val_window)
    vit_features_dir = cfg.get("cached_vit_features_dir")
    crops_dir = cfg.get("per_episode_crops_dir")
    skip_emg = cfg.get("skip_emg_loading", False)

    all_results: list[dict] = []
    base_datasets: dict[tuple, EgoEmgMemmapDataset] = {}

    for split in args.splits:
        for hand in args.hands:
            print(f"\n{'='*50}")
            print(f"Evaluating: split={split}, hand={hand}")

            dataset = EgoEmgMemmapDataset(
                memmap_dir=memmap_dir,
                window_length=int(val_window),
                stride=int(val_stride),
                allowed_splits=[split],
                modalities=["emg", "joint_angles", "labels"],
                target_hand=hand,
                emg_field_preference="filtered",
                emg_layout=emg_layout,
                emg2pose_channel_indices=channel_indices,
                channel_interpolate=channel_interpolate,
                norm_mode="per-dataset",
                norm_stats_path=norm_stats_path,
                dataset_name="egoemg",
                jitter=False,
                cached_vit_features_dir=vit_features_dir,
                per_episode_crops_dir=crops_dir,
                vision_num_frames=cfg.get("vision_num_frames", 0),
                vision_frame_selection=cfg.get("vision_frame_selection", "center"),
                vision_patch_size=cfg.get("vision_patch_size", 256),
                center_target_only=cfg.get("center_target_only", False),
                skip_emg_loading=skip_emg,
            )

            n_total = len(dataset)
            print(f"  Samples: {n_total:,}")

            rng = np.random.default_rng(args.seed)
            if args.max_samples_per_split > 0 and n_total > args.max_samples_per_split:
                indices = sorted(rng.choice(n_total, size=args.max_samples_per_split, replace=False).tolist())
            else:
                indices = list(range(n_total))

            ds_with_meta = _DatasetWithMeta(dataset, indices)
            dataloader = DataLoader(
                ds_with_meta, batch_size=args.batch_size, shuffle=False,
                num_workers=4, collate_fn=_collate_fusion, pin_memory=True,
            )

            split_results = evaluate_per_sample(module, dataloader, device, inner_model, emg_model=emg_model, vis_model=vis_model)
            for r in split_results:
                r["split"] = split
                r["hand"] = hand
            all_results.extend(split_results)
            base_datasets[(split, hand)] = dataset
            print(f"  Collected: {len(split_results)} samples")

    inner_model._forward_center_supervised = original_forward_cs

    # ── Filter to only samples with valid mocap (matching verify_training_with_emg.py) ──
    tracked_mm = _load_mm(Path(args.data_location), "mocap_webcam_tracked")
    stale_mm = _load_mm(Path(args.data_location), "image_webcam_stale")
    tracked_count = 0
    filtered_results: list[dict] = []
    for r in all_results:
        ci = r.get("center_idx")
        if ci is None:
            continue
        hand = r.get("hand", "right")
        # Must be tracked, not stale, and have valid markers for target hand
        if not (bool(tracked_mm[ci]) and not bool(stale_mm[ci])):
            continue
        valid_mm = _load_mm(Path(args.data_location), f"mocap_{hand}_valid")
        if not bool(valid_mm[ci].any()):
            continue
        tracked_count += 1
        filtered_results.append(r)
    print(f"\nFiltered: {tracked_count}/{len(all_results)} samples with valid mocap tracking")
    all_results = filtered_results
    if not all_results:
        print("ERROR: No samples with valid mocap found!")
        return

    # ── Rank ─────────────────────────────────────────────────────────────────
    all_results.sort(key=lambda x: x["delta_mae"], reverse=True)

    print(f"\n{'='*50}")
    print(f"Total: {len(all_results)} samples")
    print(f"\nTop-{args.num_samples} where fusion helps most:")
    print(f"{'Rank':<6} {'Split':<10} {'Hand':<6} {'Vis(°)':<10} {'Fus(°)':<10} {'Δ(°)':<10}")
    print("-" * 52)
    for i, r in enumerate(all_results[:args.num_samples]):
        print(f"{i+1:<6} {r['split']:<10} {r['hand']:<6} "
              f"{r['mae_vision']*57.3:<10.2f} {r['mae_fusion']*57.3:<10.2f} "
              f"{r['delta_mae']*57.3:<10.2f}")

    deltas = np.array([r["delta_mae"] for r in all_results])
    print(f"\nDelta stats ({len(deltas)} samples):")
    print(f"  Mean: {deltas.mean()*57.3:.2f}°  Median: {np.median(deltas)*57.3:.2f}°")
    print(f"  Std: {deltas.std()*57.3:.2f}°  Positive: {(deltas>0).sum()} ({(deltas>0).mean()*100:.1f}%)")
    print(f"  P10: {np.percentile(deltas,10)*57.3:.2f}°  P90: {np.percentile(deltas,90)*57.3:.2f}°")

    emg_maes = np.array([r.get("mae_emg", 0) for r in all_results])
    if (emg_maes > 0).any():
        print(f"\nModality MAE (all {len(all_results)} samples):")
        print(f"  Vision: {np.array([r['mae_vision'] for r in all_results]).mean()*57.3:.2f}°")
        print(f"  EMG:    {emg_maes.mean()*57.3:.2f}°")
        print(f"  Fusion: {np.array([r['mae_fusion'] for r in all_results]).mean()*57.3:.2f}°")

    # ── Save CSVs ────────────────────────────────────────────────────────────
    import pandas as pd
    csv_rows = [{
        "rank": i+1, "split": r["split"], "hand": r["hand"],
        "mae_vision_deg": round(r["mae_vision"]*57.3, 2),
        "mae_emg_deg": round(r.get("mae_emg", 0)*57.3, 2),
        "mae_fusion_deg": round(r["mae_fusion"]*57.3, 2),
        "delta_deg": round(r["delta_mae"]*57.3, 2),
        "ep_idx": r.get("ep_idx", -1),
        "center_idx": r.get("center_idx", -1),
    } for i, r in enumerate(all_results)]
    pd.DataFrame(csv_rows).to_csv(output_dir / "fusion_wins_ranking.csv", index=False)

    summary_rows = []
    for split in args.splits:
        for hand in args.hands:
            sub = [r for r in all_results if r["split"] == split and r["hand"] == hand]
            if not sub: continue
            d = np.array([r["delta_mae"] for r in sub]) * 57.3
            summary_rows.append({
                "split": split, "hand": hand, "n": len(sub),
                "mean_delta_deg": round(float(d.mean()), 2),
                "median_delta_deg": round(float(np.median(d)), 2),
                "pct_positive": round(float((d>0).mean()*100), 1),
                "p90_delta_deg": round(float(np.percentile(d, 90)), 2),
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / "fusion_wins_summary.csv", index=False)

    # ── Rich visualization for top-k ─────────────────────────────────────────
    n_vis = min(args.num_samples, len(all_results))
    print(f"\nGenerating rich visualizations for top {n_vis} samples...")

    # Load calibration once
    calib_path = _DATA_ROOT / "reprojection_assets" / "GX010023_standard_calibration.json"
    with open(calib_path) as f:
        calib = json.load(f)

    for i in tqdm(range(n_vis), desc="Visualizing"):
        r = all_results[i]
        # Find the right base dataset for this split/hand
        key = (r["split"], r["hand"])
        ds = base_datasets.get(key)
        if ds is None:
            ds = list(base_datasets.values())[0]
        try:
            visualize_sample(
                r, output_dir, i,
                memmap_dir=Path(args.data_location),
                video_root=Path(args.video_root),
                dataset=ds,
                device=device,
                calib=calib,
            )
        except Exception as e:
            print(f"  [ERROR] Sample {i+1} visualization failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Results saved to {output_dir}/")
    print(f"Each sample_NNN/ dir contains:")
    for f in ["original.png", "crop.png", "markers_proj_original.png",
              "mano_proj_original.png", "mano_bbox_original.png",
              "gt_skeleton_original.png", "vision_skeleton_original.png",
              "fusion_skeleton_original.png", "comparison_original.png",
              "error_comparison.png",
              "gt_from_angles.glb", "pred_vision.glb", "pred_fusion.glb", "mano_gt.glb"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
