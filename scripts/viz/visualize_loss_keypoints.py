#!/usr/bin/env python3
"""Visualize the exact 3D/2D keypoints that enter the loss function.

For the current WiLoR checkpoint (pretrained or fine-tuned), runs one
validation batch, captures pred_keypoints_3d/2d and GT keypoints_3d/2d
as they are passed into Keypoint3DLoss and Keypoint2DLoss, and saves:

  - loss_keypoints_3d_rootrel.glb: pred (red) + GT (green) root-relative skeletons
  - loss_keypoints_3d_raw.glb:     pred (red) + GT (green) raw (non-root-rel) skeletons
  - loss_keypoints_2d_{i}.jpg:     2D keypoints drawn on the input patch

Usage:
    python scripts/viz/visualize_loss_keypoints.py \
        data_location=/path/to/EgoEMG_memmap \
        video_root=/path/to/EgoEMG \
        mano_model_path=/path/to/mano_data \
        wilor_checkpoint_path=/path/to/wilor_final.ckpt
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import trimesh

MANOTORCH_ROOT = Path("/home/xiziheng/develop/manotorch")
WILOR_PATH = Path(__file__).resolve().parents[1] / ".." / "WiLoR"
for p in [MANOTORCH_ROOT, WILOR_PATH]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from emg2pose.datasets.egoemg_vision_dataset import EgoEmgVisionDataset
from emg2pose.models.wilor_egoemg import EgoEMGWiLoRModule
from wilor.utils.geometry import perspective_projection

from emg2pose.train_vision import _build_wilor_cfg

log = logging.getLogger(__name__)

# MANO 21-joint skeleton connectivity (smplx ordering).
# 16 base joints from J_regressor + 5 fingertips appended in dict order
# (thumb, index, middle, ring, pinky) from smplx/vertex_ids.py.
#   0  = wrist
#   1  = index MCP,  2  = index PIP,  3  = index DIP
#   4  = middle MCP, 5  = middle PIP, 6  = middle DIP
#   7  = pinky MCP,  8  = pinky PIP,  9  = pinky DIP
#  10  = ring MCP,  11 = ring PIP,   12 = ring DIP
#  13  = thumb CMC,  14 = thumb MCP,  15 = thumb IP
#  16  = thumb tip,  17 = index tip,  18 = middle tip
#  19  = ring tip,   20 = pinky tip
MANO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb: CMC→MCP→IP→tip
    (0, 5), (5, 6), (6, 7), (7, 8),          # index: MCP→PIP→DIP→tip
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle: MCP→PIP→DIP→tip
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring: MCP→PIP→DIP→tip
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky: MCP→PIP→DIP→tip
]

# MANO vertex indices corresponding to mocap marker positions.
MARKER_INDICES = torch.tensor([
    191, 88, 253, 708, 729, 144, 87, 295, 319, 220,
    365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673,
], dtype=torch.long)

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])

_STRING_KEYS = {
    "video_path", "episode_id", "episode_subject",
    "target_hand", "dataset_name", "bbox_source_name",
}

NUM_SAMPLES = 16


def _collate(batch):
    tensor_batch, string_batch = [], []
    for sample in batch:
        t, s = {}, {}
        for k, v in sample.items():
            if k in _STRING_KEYS:
                s[k] = v
            else:
                t[k] = v
        tensor_batch.append(t)
        string_batch.append(s)
    collated = default_collate(tensor_batch)
    for k in string_batch[0]:
        collated[k] = [s[k] for s in string_batch]
    return collated


def _batch_to(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device, non_blocking=True)
                if torch.is_tensor(vv) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


def denormalize_patch(img_tensor):
    img = img_tensor.cpu().float()
    mean = IMAGE_MEAN.view(3, 1, 1) * 255
    std = IMAGE_STD.view(3, 1, 1) * 255
    img = img * std + mean
    img = img.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


PRED_COLOR = [220, 50, 50, 255]
GT_COLOR = [50, 200, 50, 255]
PRED_COLOR_BGR = (50, 50, 220)
GT_COLOR_BGR = (0, 200, 0)


# ---------- 3D GLB helpers ----------

def _make_bone(p1, p2, radius, color):
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=length)
    cyl.visual.face_colors = color
    z = np.array([0.0, 0.0, 1.0])
    d = direction / length
    axis = np.cross(z, d)
    axis_len = np.linalg.norm(axis)
    if axis_len > 1e-8:
        angle = np.arccos(np.clip(np.dot(z, d), -1, 1))
        R = trimesh.transformations.rotation_matrix(angle, axis)
        cyl.apply_transform(R)
    cyl.apply_translation((p1 + p2) / 2)
    return cyl


def build_skeleton_meshes(joints, color, prefix, radius_bone=1.5, radius_joint=2.5):
    named_meshes = []
    for i in range(len(joints)):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius_joint)
        sphere.apply_translation(joints[i])
        sphere.visual.face_colors = color
        named_meshes.append((f"{prefix}_j{i}", sphere))
    for bone_idx, (parent, child) in enumerate(MANO_SKELETON):
        bone = _make_bone(joints[parent], joints[child], radius_bone, color)
        if bone is not None:
            named_meshes.append((f"{prefix}_bone{bone_idx}", bone))
    return named_meshes


def _add_to_scene(scene, named_meshes):
    for name, mesh in named_meshes:
        scene.add_geometry(mesh, node_name=name)


def save_glb(pred_joints, gt_joints, path):
    scene = trimesh.Scene()
    _add_to_scene(scene, build_skeleton_meshes(pred_joints, PRED_COLOR, "pred"))
    _add_to_scene(scene, build_skeleton_meshes(gt_joints, GT_COLOR, "gt"))
    scene.export(path)
    log.info("Saved %s", path)


def save_mesh_glb(pred_verts, gt_verts, faces, path):
    pred_mesh = trimesh.Trimesh(vertices=pred_verts, faces=faces, process=False)
    pred_mesh.visual.face_colors = PRED_COLOR
    gt_mesh = trimesh.Trimesh(vertices=gt_verts, faces=faces, process=False)
    gt_mesh.visual.face_colors = GT_COLOR
    scene = trimesh.Scene()
    scene.add_geometry(pred_mesh, node_name="pred_mesh")
    scene.add_geometry(gt_mesh, node_name="gt_mesh")
    scene.export(path)
    log.info("Saved %s", path)


# ---------- 2D image helpers ----------

def draw_skeleton_2d(img, kp2d, color, radius=4):
    h, w = img.shape[:2]
    pts = []
    for x, y in kp2d:
        px = int((x + 0.5) * w)
        py = int((y + 0.5) * h)
        pts.append((px, py))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img, (px, py), radius, color, -1)
    for parent, child in MANO_SKELETON:
        cv2.line(img, pts[parent], pts[child], color, 2, cv2.LINE_AA)
    return img


def draw_mesh_projection(img, verts_2d, faces, color, thickness=1):
    """Project mesh faces onto 2D image and draw wireframe."""
    h, w = img.shape[:2]
    pts = []
    for x, y in verts_2d:
        pts.append((int((x + 0.5) * w), int((y + 0.5) * h)))
    for f in faces:
        for j in range(3):
            p1, p2 = pts[f[j]], pts[f[(j + 1) % 3]]
            if (0 <= p1[0] < w and 0 <= p1[1] < h
                    and 0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    return img


# ---------- Main ----------

@hydra.main(
    config_path="../config", config_name="vision_base", version_base=None,
)
def main(config: DictConfig):
    device = torch.device(
        f"cuda:{config.get('devices', [0])[0]}"
        if torch.cuda.is_available() else "cpu"
    )
    wilor_cfg = _build_wilor_cfg(config)

    # Load model
    ckpt_path = config.get("checkpoint")
    if ckpt_path and os.path.exists(ckpt_path):
        log.info("Loading fine-tuned checkpoint: %s", ckpt_path)
        model = EgoEMGWiLoRModule.load_from_checkpoint(
            ckpt_path, cfg=wilor_cfg
        ).to(device)
    else:
        log.info("Loading pretrained WiLoR weights.")
        model = EgoEMGWiLoRModule(wilor_cfg).to(device)
        ckpt = torch.load(
            config.get("wilor_checkpoint_path"),
            map_location="cpu", weights_only=False,
        )
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Build a small dataset (first batch only)
    data_dir = Path(config.data_location)
    video_root = Path(config.get("video_root", data_dir.parent))
    allintra_root = config.get("allintra_root")
    vision_index_dir = config.get("vision_index_dir")
    calibration_path = config.get("calibration_path")
    loss_weights = config.get("loss_weights", {
        "keypoints_3d": 0.05, "keypoints_2d": 0.01,
    })

    ds_kwargs_base = dict(
        memmap_dir=data_dir,
        video_root=video_root,
        allintra_root=Path(allintra_root) if allintra_root else None,
        vision_index_dir=(
            Path(vision_index_dir) if vision_index_dir else None
        ),
        auto_build_index=bool(config.get("auto_build_vision_index", False)),
        calibration_path=(
            Path(calibration_path) if calibration_path else None
        ),
        target_hand="right",
        patch_size=int(
            config.get("patch_size", wilor_cfg.MODEL.IMAGE_SIZE)
        ),
        mean=255.0 * torch.tensor(wilor_cfg.MODEL.IMAGE_MEAN).numpy(),
        std=255.0 * torch.tensor(wilor_cfg.MODEL.IMAGE_STD).numpy(),
        aug_config=wilor_cfg.DATASETS.CONFIG,
        allowed_splits=["user"],
        stride=300,
        do_augment=False,
    )
    per_episode_crops = config.get("per_episode_crops_dir")
    if per_episode_crops:
        ds_kwargs_base["per_episode_crops_dir"] = Path(per_episode_crops)
    else:
        ds_kwargs_base["video_reader_cache_size"] = 4
        ds_kwargs_base["frame_cache_size"] = 2

    ds = EgoEmgVisionDataset(**ds_kwargs_base)
    log.info("Dataset size: %d", len(ds))

    loader = DataLoader(
        ds, batch_size=NUM_SAMPLES, shuffle=True,
        num_workers=0, collate_fn=_collate,
    )
    batch = next(iter(loader))
    batch = _batch_to(batch, device)

    with torch.no_grad():
        output = model.forward_step(batch, train=False)
        gt_kp3d, gt_kp2d, gt_verts = model._forward_gt_mano(batch)

    mano_faces = np.array(model.mano.faces)  # (F, 3)

    # Reproduce exactly what compute_loss does
    pred_kp3d = output["pred_keypoints_3d"]     # (B, 21, 3)
    pred_kp2d = output["pred_keypoints_2d"]     # (B, 21, 2)
    pred_vertices = output["pred_vertices"]     # (B, 778, 3)
    gt_kp3d_xyz = gt_kp3d                       # (B, 21, 3)
    gt_kp2d_xy = gt_kp2d                        # (B, 21, 2)

    label_valid = (
        batch["has_mano_params"]["global_orient"]
        .unsqueeze(-1).unsqueeze(-1).float()
    )
    gt_kp3d_for_loss = torch.cat(
        [gt_kp3d, label_valid.expand_as(gt_kp3d[..., :1])], dim=-1,
    )

    # Root-relative as Keypoint3DLoss does
    pred_kp3d_rel = pred_kp3d - pred_kp3d[:, 0:1]
    gt_kp3d_rel = gt_kp3d_xyz - gt_kp3d_xyz[:, 0:1]

    # Project all 778 vertices to 2D for mesh wireframe
    patch_size = int(config.get("patch_size", wilor_cfg.MODEL.IMAGE_SIZE))

    # Pred: weak-perspective projection (same as forward_step uses)
    pred_cam = output["pred_cam"]
    focal_length = output["focal_length"]
    pred_cam_t = output["pred_cam_t"]
    pred_verts_2d = perspective_projection(
        pred_vertices,
        translation=pred_cam_t.reshape(-1, 3),
        focal_length=focal_length.reshape(-1, 2) / wilor_cfg.MODEL.IMAGE_SIZE,
    ).reshape(-1, 778, 2)

    # GT: calibrated camera projection (same as _forward_gt_mano uses)
    wrist_pos_cam = batch["wrist_pos_cam"]
    gt_verts_cam = gt_verts + wrist_pos_cam.unsqueeze(1)
    gt_verts_2d_px = EgoEMGWiLoRModule._project_cam_points_to_patch(
        gt_verts_cam.detach().cpu().float().numpy(),
        batch["cam_K"].detach().cpu().float().numpy(),
        batch["cam_dist"].detach().cpu().float().numpy(),
        batch["cam_crop_params"].detach().cpu().float().numpy(),
        batch["box_center"].detach().cpu().float().numpy(),
        batch["box_size"].detach().cpu().float().numpy(),
        patch_size,
    )
    gt_verts_2d = torch.from_numpy(
        (gt_verts_2d_px / patch_size - 0.5).astype(np.float32),
    ).to(device)

    out_dir = Path(os.getcwd()) / "loss_keypoint_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = 1000  # m -> mm
    records = []

    for i in range(min(pred_kp3d.shape[0], NUM_SAMPLES)):
        tag = f"sample{i}"
        rec = {"sample": i}

        # --- 3D GLB: skeleton root-relative ---
        save_glb(
            pred_kp3d_rel[i].cpu().numpy() * scale,
            gt_kp3d_rel[i].cpu().numpy() * scale,
            str(out_dir / f"{tag}_kp3d_rootrel.glb"),
        )

        # --- 3D GLB: skeleton raw ---
        save_glb(
            pred_kp3d[i].cpu().numpy() * scale,
            gt_kp3d_xyz[i].cpu().numpy() * scale,
            str(out_dir / f"{tag}_kp3d_raw.glb"),
        )

        # --- 3D GLB: mesh comparison ---
        save_mesh_glb(
            pred_vertices[i].cpu().numpy() * scale,
            gt_verts[i].cpu().numpy() * scale,
            mano_faces,
            str(out_dir / f"{tag}_mesh.glb"),
        )

        # --- 2D image: skeleton GT (green) and Pred (red) side by side ---
        img = denormalize_patch(batch["img"][i])

        vis_gt = img.copy()
        vis_pred = img.copy()
        draw_skeleton_2d(vis_gt, gt_kp2d_xy[i].cpu().numpy(), GT_COLOR_BGR)
        draw_skeleton_2d(vis_pred, pred_kp2d[i].cpu().numpy(), PRED_COLOR_BGR)
        cv2.putText(vis_gt, "GT", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis_pred, "Pred", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        vis_kp = np.concatenate([vis_gt, vis_pred], axis=1)

        # --- 2D image: mesh wireframe projection ---
        vis_mesh_gt = img.copy()
        vis_mesh_pred = img.copy()
        draw_mesh_projection(
            vis_mesh_gt, gt_verts_2d[i].cpu().numpy(), mano_faces, GT_COLOR_BGR,
        )
        draw_mesh_projection(
            vis_mesh_pred, pred_verts_2d[i].cpu().numpy(), mano_faces, PRED_COLOR_BGR,
        )
        cv2.putText(vis_mesh_gt, "GT mesh", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis_mesh_pred, "Pred mesh", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # --- 2D image: mocap markers projected to patch ---
        gt_marker_color = (255, 255, 0)   # cyan in BGR
        pred_marker_color = (0, 165, 255)  # orange
        vis_markers_gt = img.copy()
        vis_markers_pred = img.copy()
        orig_m = batch["orig_markers_2d"][i].cpu().numpy()  # (N, 3) x,y,conf in raw video
        m_conf = orig_m[:, 2]
        # GT markers: both markers and bbox are in raw video pixel coords.
        bc = batch["box_center"][i].cpu().numpy()        # (2,) raw video
        bs = float(batch["box_size"][i].cpu().numpy())    # scalar, raw video
        s = patch_size / bs
        m_patch = (orig_m[:, :2] - bc[None, :]) * s + patch_size / 2.0
        m_norm = m_patch / patch_size - 0.5

        h_img, w_img = vis_markers_gt.shape[:2]
        n_valid = int((m_conf > 0).sum())
        for mi in range(len(m_norm)):
            if m_conf[mi] <= 0:
                continue
            mx = int((m_norm[mi, 0] + 0.5) * w_img)
            my = int((m_norm[mi, 1] + 0.5) * h_img)
            if 0 <= mx < w_img and 0 <= my < h_img:
                cv2.circle(vis_markers_gt, (mx, my), 3, gt_marker_color, -1)
        cv2.putText(vis_markers_gt, f"GT Markers (n={n_valid})", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Pred markers: sample pred mesh at MARKER_INDICES and project to 2D.
        mi_dev = MARKER_INDICES.to(pred_verts_2d.device)
        pred_marker_2d = pred_verts_2d[i, mi_dev]  # (21, 2) normalized [-0.5, 0.5]
        for mi_idx in range(pred_marker_2d.shape[0]):
            mx = int((pred_marker_2d[mi_idx, 0].item() + 0.5) * w_img)
            my = int((pred_marker_2d[mi_idx, 1].item() + 0.5) * h_img)
            if 0 <= mx < w_img and 0 <= my < h_img:
                cv2.circle(vis_markers_pred, (mx, my), 3, pred_marker_color, -1)
        cv2.putText(vis_markers_pred, "Pred Markers", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        vis_markers = np.concatenate([vis_markers_gt, vis_markers_pred], axis=1)

        vis_mesh = np.concatenate([vis_mesh_gt, vis_mesh_pred], axis=1)

        # --- Compute metrics ---
        valid_3d = gt_kp3d_for_loss[i, :, 3] > 0

        # Per-joint 3D root-relative error (mm)
        kp3d_err = (pred_kp3d_rel[i] - gt_kp3d_rel[i]).norm(dim=-1)
        rec["mpjpe_mm"] = kp3d_err[valid_3d].mean().item() * scale if valid_3d.any() else 0
        for j in range(21):
            rec[f"j{j:02d}_3d_err_mm"] = kp3d_err[j].item() * scale

        # PA-MPJPE (mm)
        v = valid_3d
        if v.sum() >= 3:
            p = pred_kp3d_rel[i, v].float()
            g = gt_kp3d_rel[i, v].float()
            p_c = p - p.mean(0)
            g_c = g - g.mean(0)
            H = p_c.T @ g_c
            U, _, Vh = torch.linalg.svd(H)
            d = torch.det(Vh.T @ U.T).sign()
            Vh[-1] = Vh[-1] * d
            R = Vh.T @ U.T
            p_aligned = p_c @ R.T + g.mean(0)
            rec["pa_mpjpe_mm"] = (p_aligned - g).norm(dim=-1).mean().item() * scale
        else:
            rec["pa_mpjpe_mm"] = 0.0

        # Per-joint 2D error (normalized)
        kp2d_err = (pred_kp2d[i] - gt_kp2d_xy[i]).norm(dim=-1)
        rec["kp2d_mean_err"] = kp2d_err.mean().item()
        for j in range(21):
            rec[f"j{j:02d}_2d_err"] = kp2d_err[j].item()

        # Vertex error (mm, root-relative)
        pred_v_rel = pred_vertices[i] - pred_kp3d[i, 0]
        gt_v_rel = gt_verts[i] - gt_kp3d_xyz[i, 0]
        rec["vert_err_mm"] = (pred_v_rel - gt_v_rel).norm(dim=-1).mean().item() * scale

        # Weighted losses (using config loss_weights)
        w3d = loss_weights.get("keypoints_3d", 0.05)
        w2d = loss_weights.get("keypoints_2d", 0.01)
        rec["loss_kp3d"] = w3d * rec["mpjpe_mm"] / scale
        rec["loss_kp2d"] = w2d * rec["kp2d_mean_err"]
        rec["loss_total"] = rec["loss_kp3d"] + rec["loss_kp2d"]

        rec["valid_joints"] = int(valid_3d.sum().item())
        records.append(rec)

        log.info(
            "%s: MPJPE=%.1fmm  PA-MPJPE=%.1fmm  2D_err=%.4f  "
            "loss_kp3d=%.6f  loss_kp2d=%.6f  total=%.6f",
            tag, rec["mpjpe_mm"], rec["pa_mpjpe_mm"], rec["kp2d_mean_err"],
            rec["loss_kp3d"], rec["loss_kp2d"], rec["loss_total"],
        )

        cv2.imwrite(str(out_dir / f"{tag}_kp2d.jpg"), vis_kp)
        cv2.imwrite(str(out_dir / f"{tag}_mesh2d.jpg"), vis_mesh)
        cv2.imwrite(str(out_dir / f"{tag}_markers2d.jpg"), vis_markers)

    # Save metrics CSV
    df = pd.DataFrame(records)
    csv_path = out_dir / "metrics.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log.info("Saved %s", csv_path)
    log.info("Loss weights: kp3d=%.4f  kp2d=%.4f", w3d, w2d)

    print(df[["sample", "mpjpe_mm", "pa_mpjpe_mm", "kp2d_mean_err",
              "loss_kp3d", "loss_kp2d", "loss_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
