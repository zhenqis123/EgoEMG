#!/usr/bin/env python3
"""Visualize Manus-to-UmeTrack IK fit: compare Manus target skeleton (blue)
vs UmeTrack FK skeleton (red) side by side in GLB.

Usage:
  python scripts/viz/viz_manus_fit.py \
    --session data/data/data_20260525_180032 \
    --output data/manus_fit_viz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

_SCRIPT_DIR = Path(__file__).resolve().parent
_IK_DIR = _SCRIPT_DIR.parent / "ik"
if str(_IK_DIR) not in sys.path:
    sys.path.insert(0, str(_IK_DIR))

from umetrack_fk_utils import (
    LM_WEIGHTS,
    MANUS_TO_UMETRACK_MAP,
    angles_20d_to_22d,
    extract_manus_targets,
    fk_landmarks,
    get_joint_limits,
    load_model,
    make_wrist_transform,
)

UMETRACK_LM_NAMES = [
    "L0:thumb_tip", "L1:index_tip", "L2:middle_tip", "L3:ring_tip", "L4:pinky_tip",
    "L5:wrist", "L6:thumb_root", "L7:thumb_dip",
    "L8:index_root", "L9:index_pip", "L10:index_dip",
    "L11:middle_root", "L12:middle_pip", "L13:middle_dip",
    "L14:ring_root", "L15:ring_pip", "L16:ring_dip",
    "L17:pinky_root", "L18:pinky_pip", "L19:pinky_dip",
    "L20:wrist2",
]

MANUS_NODE_NAMES = {
    0: "Wrist", 1: "Thumb_MCP", 2: "Thumb_PIP", 3: "Thumb_DIP", 4: "Thumb_TIP",
    5: "Index_MCP", 6: "Index_PIP", 7: "Index_IP", 8: "Index_DIP", 9: "Index_TIP",
    10: "Middle_MCP", 11: "Middle_PIP", 12: "Middle_IP", 13: "Middle_DIP", 14: "Middle_TIP",
    15: "Ring_MCP", 16: "Ring_PIP", 17: "Ring_IP", 18: "Ring_DIP", 19: "Ring_TIP",
    20: "Pinky_MCP", 21: "Pinky_PIP", 22: "Pinky_IP", 23: "Pinky_DIP", 24: "Pinky_TIP",
}

# Manus 25-node bone connections (from data/data/README.md)
MANUS_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),                       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),               # index
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),      # middle
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),      # ring
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),      # pinky
]

# UmeTrack 21-landmark bone connections
# lm indices: 0-4=fingertips, 5=wrist, 6-7=thumb, 8-10=index,
#              11-13=middle, 14-16=ring, 17-19=pinky, 20=wrist2
UMETRACK_BONES = [
    (5, 6), (6, 7), (7, 0),                                # thumb
    (5, 8), (8, 9), (9, 10), (10, 1),                      # index
    (5, 11), (11, 12), (12, 13), (13, 2),                  # middle
    (5, 14), (14, 15), (15, 16), (16, 3),                  # ring
    (5, 17), (17, 18), (18, 19), (19, 4),                  # pinky
]


def _rodrigues_z_to_dir(direction: np.ndarray) -> np.ndarray:
    """Rodrigues rotation: Z-axis [0,0,1] → direction (3x3 matrix)."""
    z = np.array([0.0, 0.0, 1.0])
    d = direction / np.linalg.norm(direction)
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if c > 0.99999:
        return np.eye(3)
    if c < -0.99999:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _add_cylinder_bone(meshes, pa, pb, radius, color):
    """Add a cylinder between pa and pb to the meshes list."""
    direction = pb - pa
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return
    mid = (pa + pb) / 2.0
    R = _rodrigues_z_to_dir(direction)
    cyl = trimesh.primitives.Cylinder(radius=radius, height=dist, sections=8)
    verts = cyl.vertices.copy() @ R.T + mid
    mesh = trimesh.Trimesh(vertices=verts, faces=cyl.faces, process=False)
    mesh.visual.vertex_colors = np.tile(
        np.array(color, dtype=np.uint8), (len(verts), 1)
    )
    meshes.append(mesh)


def fk_mesh(angles_20d, wrist_aa, wrist_trans, scale, hand_model):
    """Compute UmeTrack skinned mesh from angles + wrist rotation + translation + scale."""
    from emg2pose.kinematics import apply_to_hand_model, broadcast_hand_model_to
    from emg2pose.UmeTrack.lib.common.hand_skinning import (
        _get_skinned_vertices, _hand_skinning_transform, _lbs,
    )
    hm = broadcast_hand_model_to(hand_model, (1,))
    hm = apply_to_hand_model(hm, lambda t: t.float())
    wrist_tf = make_wrist_transform(wrist_aa, wrist_trans).unsqueeze(0)
    a = angles_20d_to_22d(angles_20d).reshape(1, -1)
    skin_xfs = _hand_skinning_transform(
        hm.joint_rotation_axes.reshape(1, -1, 3),
        hm.joint_rest_positions.reshape(1, -1, 3),
        a, wrist_tf,
    )
    w = hm.dense_bone_weights.reshape(1, -1, 17)
    mr = hm.mesh_vertices.reshape(1, -1, 3)
    v = _get_skinned_vertices(mr, w)
    mesh = _lbs(skin_xfs, v)[..., :3][0] * scale
    return mesh


def save_skeleton_glb(
    path: Path,
    pts_a: np.ndarray,      # Manus targets (20, 3) in UmeTrack frame (blue)
    bones_a: list,
    pts_b: np.ndarray,      # UmeTrack FK landmarks (20, 3) (red)
    bones_b: list,
    mesh_v: np.ndarray | None = None,    # UmeTrack FK mesh vertices (red)
    mesh_f: np.ndarray | None = None,    # UmeTrack FK mesh faces
):
    """Save GLB with blue=Manus targets, red=UmeTrack FK skeleton+mesh, green=correspondence."""
    meshes = []

    # Red UmeTrack FK mesh (translucent, underneath skeleton)
    if mesh_v is not None and mesh_f is not None:
        m = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f, process=False)
        m.visual.vertex_colors = np.tile(
            np.array([220, 80, 60, 80], dtype=np.uint8), (len(mesh_v), 1)
        )
        meshes.append(m)

    # Blue Manus target landmarks (20 landmarks, L0-L19)
    for i, p in enumerate(pts_a):
        s = trimesh.primitives.Sphere(center=p, radius=3.0, subdivisions=2)
        s.visual.vertex_colors = np.tile(
            np.array([70, 130, 220, 255], dtype=np.uint8), (len(s.vertices), 1)
        )
        meshes.append(s)
    for i, j in bones_a:
        _add_cylinder_bone(meshes, pts_a[i], pts_a[j], 1.3, (70, 130, 220, 255))

    # Red UmeTrack FK landmarks (20 landmarks, L0-L19)
    for i, p in enumerate(pts_b):
        s = trimesh.primitives.Sphere(center=p, radius=3.0, subdivisions=2)
        s.visual.vertex_colors = np.tile(
            np.array([220, 80, 60, 255], dtype=np.uint8), (len(s.vertices), 1)
        )
        meshes.append(s)
    for i, j in bones_b:
        _add_cylinder_bone(meshes, pts_b[i], pts_b[j], 1.3, (220, 80, 60, 255))

    # Green correspondence lines between paired landmarks
    for i in range(len(pts_a)):
        if np.allclose(pts_a[i], pts_b[i], atol=0.5):
            continue
        _add_cylinder_bone(meshes, pts_a[i], pts_b[i], 0.6, (80, 200, 80, 180))

    trimesh.Scene(meshes).export(str(path))


def optimize_single_frame(targets, hand_model, init_angles, init_wrist_aa, init_scale, init_trans, lower, upper):
    """L-BFGS: optimize 20D angles + 3D wrist rotation + scale + 3D translation."""
    params = torch.cat([
        init_angles.detach().clone(),
        init_wrist_aa.detach().clone(),
        torch.tensor([init_scale]),
        init_trans.detach().clone(),
    ])
    params.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [params], lr=0.1, max_iter=100, history_size=20,
        line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        with torch.no_grad():
            params[:20].clamp_(lower, upper)
            params[23].clamp_(0.3, 2.0)
            params[24:27].clamp_(-50, 50)
        full = angles_20d_to_22d(params[:20])
        wrist_tf = make_wrist_transform(params[20:23], params[24:27])
        lm = fk_landmarks(full, hand_model, wrist_transform=wrist_tf)[:20] * params[23]
        loss = ((lm - targets).pow(2).sum(dim=-1) * LM_WEIGHTS).mean()
        (loss + 1e-5 * params[:20].pow(2).sum()).backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        params[:20].clamp_(lower, upper)
        params[23].clamp_(0.3, 2.0)
        params[24:27].clamp_(-50, 50)
    return (params[:20].detach(), params[20:23].detach(), params[23].item(),
            params[24:27].detach(), closure().item())


def main():
    parser = argparse.ArgumentParser(
        description="Compare Manus skeleton vs UmeTrack FK skeleton in GLB"
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", default="data/manus_fit_viz")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    session_dir = Path(args.session).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_name = session_dir.name

    right_jsonl = session_dir / "manus_right.jsonl"
    if not right_jsonl.exists():
        print(f"No {right_jsonl}")
        return
    frames = []
    with open(right_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    print(f"Loaded {len(frames)} Manus right frames from {session_name}")

    hand_model = load_model()
    ut_faces = hand_model.mesh_triangles.numpy()
    lower, upper = get_joint_limits(hand_model)

    first_kp = np.array(frames[0]["keypoints"], dtype=np.float32) * 1000
    scale = float(
        np.linalg.norm(first_kp[14] - first_kp[0])
        / np.linalg.norm(
            hand_model.landmark_rest_positions.numpy()[2]
            - hand_model.landmark_rest_positions.numpy()[5]
        )
    )
    print(f"Scale: {scale:.4f}")

    n_frames = len(frames)
    anchor_indices = list(range(0, n_frames, args.step))
    if anchor_indices[-1] != n_frames - 1:
        anchor_indices.append(n_frames - 1)
    anchor_set = set(anchor_indices)

    viz_frames = min(args.num_frames, n_frames)
    prev_angles = torch.zeros(20)
    prev_wrist_aa = torch.full((3,), 1e-4)
    prev_scale = scale
    prev_trans = torch.zeros(3)

    for fi in range(viz_frames):
        kp_manus = np.array(frames[fi]["keypoints"], dtype=np.float32) * 1000
        targets = extract_manus_targets(kp_manus)  # (20,3) in UmeTrack frame

        if fi in anchor_set:
            angles, wrist_aa, opt_scale, opt_trans, loss = optimize_single_frame(
                targets, hand_model, prev_angles.clone(), prev_wrist_aa.clone(),
                prev_scale, prev_trans.clone(), lower, upper,
            )
            prev_angles = angles.clone()
            prev_wrist_aa = wrist_aa.clone()
            prev_scale = opt_scale
            prev_trans = opt_trans.clone()
        else:
            angles = prev_angles.clone()
            wrist_aa = prev_wrist_aa.clone()
            opt_scale = prev_scale
            opt_trans = prev_trans.clone()
            wrist_tf = make_wrist_transform(wrist_aa, opt_trans)
            full = angles_20d_to_22d(angles)
            lm = fk_landmarks(full, hand_model, wrist_transform=wrist_tf)[:20] * opt_scale
            loss = ((lm - targets).pow(2).sum(dim=-1) * LM_WEIGHTS).mean().item()

        # FK landmarks from fitted angles (20 landmarks, L0-L19)
        with torch.no_grad():
            wrist_tf = make_wrist_transform(wrist_aa, opt_trans)
            full = angles_20d_to_22d(angles)
            fk_lm = fk_landmarks(full, hand_model, wrist_transform=wrist_tf)[:20] * opt_scale
            fk_lm_np = fk_lm.numpy()

        targets_np = targets.numpy()

        # Per-landmark errors
        per_lm_err = np.linalg.norm(targets_np - fk_lm_np, axis=1)
        print(f"\nFrame {fi} — loss={loss:.4f}, scale={opt_scale:.4f}")
        for lm_idx in range(20):
            manus_node = MANUS_TO_UMETRACK_MAP[lm_idx]
            node_name = MANUS_NODE_NAMES.get(manus_node, f"node{manus_node}")
            print(f"  {UMETRACK_LM_NAMES[lm_idx]:25s} → Manus {manus_node:2d} "
                  f"({node_name:15s}) {per_lm_err[lm_idx]:8.2f} mm")

        # FK mesh
        with torch.no_grad():
            ut_mesh_v = fk_mesh(angles, wrist_aa, opt_trans, opt_scale, hand_model).numpy()

        glb_path = output_dir / f"{session_name}_frame_{fi:04d}_fit.glb"
        save_skeleton_glb(glb_path, targets_np, UMETRACK_BONES,
                          fk_lm_np, UMETRACK_BONES,
                          mesh_v=ut_mesh_v, mesh_f=ut_faces)
        print(f"  Saved: {glb_path}")

    print(f"\nDone. Output in {output_dir}/")


if __name__ == "__main__":
    main()
