#!/usr/bin/env python3
"""Visualize IK quality: compare MANO mesh vs UmeTrack FK mesh for a single frame.
Exports a combined GLB with both meshes side-by-side.

Usage:
    PYOPENGL_PLATFORM=osmesa python scripts/viz/viz_ik_compare.py \
        --memmap-root data/sess_20260530_140912/memmap \
        --frame 500 --hand right
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

MANOTORCH_ROOT = Path("/home/xiziheng/develop/manotorch")
if str(MANOTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MANOTORCH_ROOT))
from manotorch.manolayer import ManoLayer

MANO_ASSETS_ROOT = Path("/home/xiziheng/develop/HandVQVAE/assets/mano")
ALIGN_SCALE = 1.0843137502670288
ALIGN_TRANS = np.array([106.72334, -11.8804455, -4.48328], dtype=np.float32)


def umetrack_fk(hand_model, angles_22, device):
    from emg2pose.kinematics import broadcast_hand_model_to, apply_to_hand_model
    from emg2pose.UmeTrack.lib.common.hand_skinning import (
        _get_skinned_vertices, _hand_skinning_transform, _lbs,
    )
    hm = broadcast_hand_model_to(hand_model, (1,))
    hm = apply_to_hand_model(hm, lambda t: t.float().to(device))
    wrist_tf = torch.eye(4, device=device).unsqueeze(0)
    skin_xfs = _hand_skinning_transform(
        hm.joint_rotation_axes.reshape(1, -1, 3),
        hm.joint_rest_positions.reshape(1, -1, 3),
        angles_22.unsqueeze(0), wrist_tf,
    )
    w = hm.dense_bone_weights.reshape(1, -1, 17)
    mr = hm.mesh_vertices.reshape(1, -1, 3)
    v = _get_skinned_vertices(mr, w)
    mesh = _lbs(skin_xfs, v)[0, :, :3].cpu().numpy()
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--output", type=Path, default=Path("ik_compare_output"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    side = args.hand
    args.output.mkdir(parents=True, exist_ok=True)

    # Load memmap
    with open(args.memmap_root / "manifest.json") as f:
        manifest = json.load(f)
    pose_info = manifest["fields"][f"generated_mano_{side}_pose"]
    pose_mm = np.memmap(
        args.memmap_root / pose_info["filename"],
        dtype=pose_info["dtype"], mode="r",
        shape=tuple(pose_info["shape"]),
    )
    fi = args.frame
    pose_np = pose_mm[fi:fi+1].astype(np.float32).copy()
    print(f"Frame {fi}/{pose_mm.shape[0]}")

    # Also load saved UmeTrack angles if available
    ut_angles = None
    ja_path = args.memmap_root / f"generated_joint_angles_{side}.dat"
    if ja_path.exists():
        ja_mm = np.memmap(ja_path, dtype=np.float32, mode="r",
                          shape=(manifest["total_rows"], 20))
        ut_angles = ja_mm[fi].copy()
        print(f"Loaded saved UmeTrack angles: {ut_angles}")

    # Set up MANO
    mano_layer = ManoLayer(
        rot_mode="axisang", side="right",
        mano_assets_root=str(MANO_ASSETS_ROOT),
        use_pca=False, flat_hand_mean=False,
    ).to(device)
    pose_t = torch.from_numpy(pose_np).to(device)
    beta_t = torch.zeros(1, 10, device=device)

    with torch.no_grad():
        out = mano_layer(pose_t, beta_t)
    mano_verts = (out.verts * 1000.0)[0].cpu().numpy()
    mano_faces = mano_layer.th_faces.cpu().numpy()

    # Set up UmeTrack
    from emg2pose.kinematics import apply_to_hand_model, load_default_hand_model
    hand_model = load_default_hand_model()
    hand_model = apply_to_hand_model(hand_model, lambda t: t.float().to(device))
    ut_faces = hand_model.mesh_triangles.cpu().numpy()[:, ::-1]

    # UmeTrack rest mesh
    ut_rest = umetrack_fk(hand_model, torch.zeros(22, device=device), device)

    # UmeTrack FK from saved angles
    ut_opt_mesh = None
    if ut_angles is not None and ut_angles.sum() != 0:
        ut_opt_mesh = umetrack_fk(
            hand_model,
            torch.from_numpy(ut_angles).float().to(device),
            device,
        )
        print(f"UmeTrack optimized mesh span: {ut_opt_mesh.max(0) - ut_opt_mesh.min(0)}")

    scale = ALIGN_SCALE
    trans = ALIGN_TRANS
    print(f"Alignment: scale={scale:.4f}, trans={trans}")

    # Align UmeTrack to MANO (UmeTrack is left-hand frame, negate X for right)
    flip = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

    def align(verts):
        return (scale * (verts @ flip)) + trans

    ut_rest_aligned = align(ut_rest)
    if ut_opt_mesh is not None:
        ut_opt_aligned = align(ut_opt_mesh)

    # Save individual GLBs
    def save_glb(verts, faces, color, path):
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        m.visual.vertex_colors = list(color) + [200]
        m.export(str(path))

    save_glb(mano_verts, mano_faces, (100, 149, 237),
             args.output / f"mano_{side}_frame{fi:06d}.glb")
    save_glb(ut_rest_aligned, ut_faces, (255, 80, 80),
             args.output / f"ut_rest_{side}_frame{fi:06d}.glb")
    if ut_opt_mesh is not None:
        save_glb(ut_opt_aligned, ut_faces, (255, 180, 0),
                 args.output / f"ut_ik_{side}_frame{fi:06d}.glb")

    # Combined scene: MANO (blue) + UmeTrack IK (orange) side-by-side offset
    mano_offset = mano_verts + np.array([0.12, 0, 0], dtype=np.float32)
    ut_offset = ut_opt_aligned if ut_opt_mesh is not None else ut_rest_aligned
    ut_offset = ut_offset - np.array([0.12, 0, 0], dtype=np.float32)

    m1 = trimesh.Trimesh(vertices=mano_offset, faces=mano_faces, process=False)
    m1.visual.vertex_colors = [100, 149, 237, 200]
    m2 = trimesh.Trimesh(vertices=ut_offset, faces=ut_faces, process=False)
    m2.visual.vertex_colors = [255, 180, 0, 200]
    scene = trimesh.Scene([m1, m2])
    combined_path = args.output / f"compare_{side}_frame{fi:06d}.glb"
    scene.export(str(combined_path))

    print(f"\nSaved to {args.output.resolve()}:")
    for f in sorted(args.output.iterdir()):
        if f.suffix == '.glb':
            print(f"  {f.name}")

    if ut_opt_mesh is not None:
        # Use chamfer distance since vertex counts differ
        from scipy.spatial import cKDTree
        tree_a = cKDTree(mano_verts)
        tree_b = cKDTree(ut_opt_aligned)
        d_ab, _ = tree_a.query(ut_opt_aligned)
        d_ba, _ = tree_b.query(mano_verts)
        chamfer = 0.5 * (d_ab.mean() + d_ba.mean())
        print(f"\nChamfer distance (MANO vs UmeTrack IK): {chamfer:.3f} mm")


if __name__ == "__main__":
    main()
