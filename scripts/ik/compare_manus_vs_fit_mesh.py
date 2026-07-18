#!/usr/bin/env python3
"""Compare UmeTrack FK meshes from Manus native angles vs fitted UmeTrack angles.

For the same frames, render both meshes side-by-side to visualize the angle
space mismatch.

Usage:
  python scripts/ik/compare_manus_vs_fit_mesh.py \
    --session data/data/data_20260526_132327 \
    --fit-dir data/manus_fit/data_20260526_132327 \
    --output data/mesh_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from emg2pose.kinematics import apply_to_hand_model, load_default_hand_model
from emg2pose.UmeTrack.lib.common.hand_skinning import (
    _get_skinned_vertices,
    _hand_skinning_transform,
    _lbs,
    skin_landmarks,
)

# UmeTrack FK landmark → skeleton connectivity (L0-L19, 20 landmarks).
# Landmark layout:
#   0-4:  fingertips (thumb, index, middle, ring, pinky)
#   5:    wrist
#   6-7:  thumb (CMC, DIP)
#   8-10: index (CMC, MCP, IP)
#   11-13: middle (CMC, MCP, IP)
#   14-16: ring (CMC, MCP, IP)
#   17-19: pinky (CMC, MCP, IP)
UMETRACK_SKELETON_EDGES = [
    # wrist → finger CMCs
    (5, 6), (5, 8), (5, 11), (5, 14), (5, 17),
    # thumb: CMC → DIP → tip
    (6, 7), (7, 0),
    # index: CMC → MCP → IP → tip
    (8, 9), (9, 10), (10, 1),
    # middle: CMC → MCP → IP → tip
    (11, 12), (12, 13), (13, 2),
    # ring: CMC → MCP → IP → tip
    (14, 15), (15, 16), (16, 3),
    # pinky: CMC → MCP → IP → tip
    (17, 18), (18, 19), (19, 4),
]

FINGER_COLORS = [
    [70, 130, 180, 255],    # thumb: blue
    [220, 80, 60, 255],     # index: red
    [60, 180, 120, 255],    # middle: green
    [230, 180, 40, 255],    # ring: orange
    [160, 80, 200, 255],    # pinky: purple
]

# Which finger each bone belongs to (index into UMETRACK_SKELETON_EDGES)
_BONE_FINGER = (
    [0] * 5   # wrist→CMC bones all belong to thumb-finger (we color them per dest)
    + [0, 0]   # thumb
    + [1, 1, 1]  # index
    + [2, 2, 2]  # middle
    + [3, 3, 3]  # ring
    + [4, 4, 4]  # pinky
)


def umetrack_fk(hand_model, angles_22, device):
    """UmeTrack FK: 22D angles -> landmarks (21,3) + mesh verts."""
    from emg2pose.kinematics import broadcast_hand_model_to

    hm = broadcast_hand_model_to(hand_model, (1,))
    hm = apply_to_hand_model(hm, lambda t: t.float())
    wrist_tf = torch.eye(4, device=device).unsqueeze(0)
    a = angles_22.reshape(1, -1)
    lm = skin_landmarks(hm, a[:, :20], wrist_tf)[0]
    skin_xfs = _hand_skinning_transform(
        hm.joint_rotation_axes.reshape(1, -1, 3),
        hm.joint_rest_positions.reshape(1, -1, 3),
        a,
        wrist_tf,
    )
    w = hm.dense_bone_weights.reshape(1, -1, 17)
    mr = hm.mesh_vertices.reshape(1, -1, 3)
    v = _get_skinned_vertices(mr, w)
    mesh = _lbs(skin_xfs, v)[..., :3][0]
    return lm, mesh


ANGLE_NAMES = [
    "thumb_cmc_aa", "thumb_cmc_fe", "thumb_mcp_fe", "thumb_ip_fe",
    "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
    "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
    "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
    "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
]


def load_manus_angles(jsonl_path: Path, frame_indices: list[int]) -> np.ndarray:
    """Load Manus native angles (degrees) for given frame indices. Returns (N, 20) rad."""
    angles_deg = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i in set(frame_indices):
                d = json.loads(line)
                angles_deg.append([d["angles"][n] for n in ANGLE_NAMES])
            if i > max(frame_indices):
                break
    return np.deg2rad(np.array(angles_deg, dtype=np.float32))


def load_fit_angles(npy_path: Path, frame_indices: list[int]) -> np.ndarray:
    """Load fitted UmeTrack angles for given frame indices. Returns (N, 20) rad."""
    data = np.load(npy_path)  # (T, 20) float32 rad
    return np.array(data[frame_indices], dtype=np.float32)


# ── Mesh helpers ──

def _make_bone(p1, p2, radius, color):
    """Create a cylinder mesh between two 3D points."""
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


def build_skeleton_meshes(landmarks, bone_radius=1.2, joint_radius=2.0):
    """Build joint spheres + bone cylinders from UmeTrack 20-landmark array."""
    meshes = []
    lm = landmarks[:20]  # L0-L19 only, skip L20 wrist2

    # Joint spheres, colored by finger
    joint_finger_map = [0, 1, 2, 3, 4, -1, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    for i in range(20):
        fc = FINGER_COLORS[joint_finger_map[i]] if joint_finger_map[i] >= 0 else [180, 180, 180, 255]
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=joint_radius)
        sphere.apply_translation(lm[i])
        sphere.visual.face_colors = fc
        meshes.append(sphere)

    # Bone cylinders
    for bi, (p, c) in enumerate(UMETRACK_SKELETON_EDGES):
        fc = FINGER_COLORS[_BONE_FINGER[bi]]
        bone = _make_bone(lm[p], lm[c], bone_radius, fc)
        if bone is not None:
            meshes.append(bone)

    return meshes


def save_mesh_glb(path, verts, faces, color):
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.visual.vertex_colors = np.tile(color, (len(verts), 1)).astype(np.uint8)
    m.export(str(path))


def save_combined_glb(path, verts_a, faces_a, verts_b, faces_b):
    """Two meshes + skeletons in one GLB. A=blue, B=red."""
    ma = trimesh.Trimesh(vertices=verts_a, faces=faces_a, process=False)
    ma.visual.vertex_colors = np.tile([70, 130, 180, 200], (len(verts_a), 1)).astype(np.uint8)

    mb = trimesh.Trimesh(vertices=verts_b, faces=faces_b, process=False)
    mb.visual.vertex_colors = np.tile([220, 80, 60, 200], (len(verts_b), 1)).astype(np.uint8)

    scene = trimesh.Scene([ma, mb])
    scene.export(str(path))


def save_skeleton_glb(path, landmarks, label_prefix=""):
    """Save skeleton (joints + bones) as GLB."""
    scene = trimesh.Scene()
    for m in build_skeleton_meshes(landmarks):
        scene.add_geometry(m)
    scene.export(str(path))


def save_mesh_with_skeleton_glb(path, verts, faces, landmarks, mesh_color):
    """Save a single mesh with its skeleton overlaid."""
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.visual.vertex_colors = np.tile(mesh_color, (len(verts), 1)).astype(np.uint8)
    # Semi-transparent mesh via face colors with alpha
    m.visual.face_colors = np.tile(mesh_color, (len(faces), 1)).astype(np.uint8)

    scene = trimesh.Scene([m])
    for sm in build_skeleton_meshes(landmarks):
        scene.add_geometry(sm)
    scene.export(str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Compare UmeTrack FK meshes: Manus native vs fitted angles"
    )
    parser.add_argument("--session", default="data/data/data_20260526_132327")
    parser.add_argument("--fit-dir", default="data/manus_fit/data_20260526_132327")
    parser.add_argument("--output", default="data/mesh_compare")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    jsonl_path = Path(args.session) / "manus_right.jsonl"
    fit_npy = Path(args.fit_dir) / "joint_angles_right.npy"

    # Pick frames
    fit_data = np.load(fit_npy)
    n_total = min(len(fit_data), 50)
    indices = list(range(0, n_total, args.stride))[:args.num_frames]
    print(f"Frames: {indices}")

    # Load angles
    manus_angles = load_manus_angles(jsonl_path, indices)
    fit_angles = load_fit_angles(fit_npy, indices)

    print(f"Manus native angles: {manus_angles.shape}")
    print(f"Fitted angles:       {fit_angles.shape}")

    # Load hand model
    print("Loading UmeTrack hand model...")
    hand_model = load_default_hand_model()
    hand_model = apply_to_hand_model(hand_model, lambda t: t.float().to(device))
    faces = hand_model.mesh_triangles.cpu().numpy()

    # FK + export per frame
    for j, idx in enumerate(indices):
        print(f"\n--- Frame {idx} ---")

        manus_22 = torch.from_numpy(
            np.concatenate([manus_angles[j], [0, 0]], axis=0)
        ).float().to(device)
        with torch.no_grad():
            manus_lm, manus_mesh = umetrack_fk(hand_model, manus_22, device)

        fit_22 = torch.from_numpy(
            np.concatenate([fit_angles[j], [0, 0]], axis=0)
        ).float().to(device)
        with torch.no_grad():
            fit_lm, fit_mesh = umetrack_fk(hand_model, fit_22, device)

        manus_v = manus_mesh.cpu().numpy()
        fit_v = fit_mesh.cpu().numpy()
        manus_lm_np = manus_lm.cpu().numpy()
        fit_lm_np = fit_lm.cpu().numpy()

        # Per-angle diff
        print(f"  {'Angle':<20s} {'Manus(°)':>10s} {'Fit(°)':>10s} {'Diff(°)':>10s}")
        print(f"  {'-'*50}")
        for ai, name in enumerate(ANGLE_NAMES):
            m = np.rad2deg(manus_angles[j, ai])
            f = np.rad2deg(fit_angles[j, ai])
            d = abs(m - f)
            print(f"  {name:<20s} {m:>10.1f} {f:>10.1f} {d:>10.1f}")

        # Combined mesh comparison
        save_combined_glb(
            out_dir / f"frame_{idx:04d}.glb",
            manus_v, faces, fit_v, faces,
        )
        # Individual mesh + skeleton
        save_mesh_with_skeleton_glb(
            out_dir / f"frame_{idx:04d}_manus.glb",
            manus_v, faces, manus_lm_np,
            [70, 130, 180, 200],
        )
        save_mesh_with_skeleton_glb(
            out_dir / f"frame_{idx:04d}_fit.glb",
            fit_v, faces, fit_lm_np,
            [220, 80, 60, 200],
        )
        # Skeleton-only
        save_skeleton_glb(out_dir / f"frame_{idx:04d}_manus_skel.glb", manus_lm_np)
        save_skeleton_glb(out_dir / f"frame_{idx:04d}_fit_skel.glb", fit_lm_np)

    print(f"\nDone. Output in {out_dir}/")
    print("  blue=Manus native angles, red=Fitted UmeTrack angles")
    print("  *_manus.glb / *_fit.glb: mesh + skeleton")
    print("  *_manus_skel.glb / *_fit_skel.glb: skeleton only")


if __name__ == "__main__":
    main()
