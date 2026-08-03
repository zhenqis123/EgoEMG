#!/usr/bin/env python3
"""Compare UmeTrack FK mesh from EgoEMG (MANO) vs Manus (UmeTrack) joint angles.

For each dataset, pick a few frames, run UmeTrack forward kinematics on the 20D
joint angles, and export a GLB with both meshes side-by-side.

Usage:
  python scripts/ik/compare_fk_mesh.py --output data/fk_mesh_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent

from egoemg.kinematics import apply_to_hand_model, load_default_hand_model
from egoemg.UmeTrack.lib.common.hand_skinning import (
    _get_skinned_vertices,
    _hand_skinning_transform,
    _lbs,
    skin_landmarks,
)


def umetrack_fk(hand_model, angles_22, device):
    """UmeTrack FK: 22D angles → landmarks (21,3) + mesh verts."""
    from egoemg.kinematics import broadcast_hand_model_to

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


def save_combined_glb(path, verts_a, faces_a, verts_b, faces_b,
                      label_a="A", label_b="B"):
    """Save two meshes in one GLB. A=blue, B=red."""
    import trimesh

    ma = trimesh.Trimesh(vertices=verts_a, faces=faces_a[:, ::-1], process=False)
    ma.visual.vertex_colors = np.tile(
        [70, 130, 180, 200], (len(verts_a), 1)
    ).astype(np.uint8)

    mb = trimesh.Trimesh(vertices=verts_b, faces=faces_b, process=False)
    mb.visual.vertex_colors = np.tile(
        [220, 80, 60, 200], (len(verts_b), 1)
    ).astype(np.uint8)

    scene = trimesh.Scene([ma, mb])
    scene.export(str(path))
    print(f"  Saved {path}  (blue={label_a}, red={label_b})")


def load_angles_from_memmap(memmap_dir: Path, num_frames: int = 5,
                            stride: int = 10000) -> np.ndarray:
    """Load joint angles from a memmap dataset.

    Returns (num_frames, 20) float32 array.
    """
    ja_path = memmap_dir / "generated_joint_angles_right.dat"
    ja = np.memmap(str(ja_path), dtype=np.float32, mode="r")
    ja = ja.reshape(-1, 20)

    indices = list(range(0, min(len(ja), stride * num_frames), stride))[:num_frames]
    return np.array(ja[indices], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Compare UmeTrack FK meshes from different angle sources"
    )
    parser.add_argument("--egoemg-dir", default="data/EgoEMG_memmap")
    parser.add_argument("--manus-dir", default="data/manus_memmap")
    parser.add_argument("--output", default="data/fk_mesh_compare")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load hand model ──
    print("Loading UmeTrack hand model...")
    hand_model = load_default_hand_model()
    hand_model = apply_to_hand_model(hand_model, lambda t: t.float().to(device))
    faces = hand_model.mesh_triangles.cpu().numpy()

    # ── Load angles from both datasets ──
    egoemg_angles = load_angles_from_memmap(
        Path(args.egoemg_dir), args.num_frames, args.stride
    )
    manus_angles = load_angles_from_memmap(
        Path(args.manus_dir), args.num_frames, args.stride
    )

    print(f"\nEgoEMG angles: {egoemg_angles.shape}")
    print(f"  Per-dim mean: {np.round(egoemg_angles.mean(axis=0), 3)}")
    print(f"  Per-dim std:  {np.round(egoemg_angles.std(axis=0), 3)}")

    print(f"\nManus angles: {manus_angles.shape}")
    print(f"  Per-dim mean: {np.round(manus_angles.mean(axis=0), 3)}")
    print(f"  Per-dim std:  {np.round(manus_angles.std(axis=0), 3)}")

    # ── FK + export per frame ──
    angle_names = [
        "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe",
        "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
        "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
        "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
        "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
    ]

    for i in range(args.num_frames):
        print(f"\n--- Frame {i} ---")

        # EgoEMG (MANO angles) → UmeTrack FK
        ego_22 = torch.from_numpy(
            np.concatenate([egoemg_angles[i], [0, 0]], axis=0)
        ).float().to(device)
        with torch.no_grad():
            ego_lm, ego_mesh = umetrack_fk(hand_model, ego_22, device)

        # Manus (UmeTrack angles) → UmeTrack FK
        manus_22 = torch.from_numpy(
            np.concatenate([manus_angles[i], [0, 0]], axis=0)
        ).float().to(device)
        with torch.no_grad():
            manus_lm, manus_mesh = umetrack_fk(hand_model, manus_22, device)

        print(f"  EgoEMG angles (deg):  {[f'{v*180/np.pi:.1f}' for v in egoemg_angles[i]]}")
        print(f"  Manus angles (deg):   {[f'{v*180/np.pi:.1f}' for v in manus_angles[i]]}")
        print(f"  Per-angle diff (deg): {[f'{(m-e)*180/np.pi:.1f}' for m, e in zip(manus_angles[i], egoemg_angles[i])]}")

        # Export individual GLBs per source
        ego_verts = ego_mesh.cpu().numpy()
        manus_verts = manus_mesh.cpu().numpy()

        save_combined_glb(
            out_dir / f"frame_{i:02d}.glb",
            ego_verts, faces, manus_verts, faces,
            label_a="EgoEMG (MANO angles)", label_b="Manus (UmeTrack angles)",
        )

        # Also save individual meshes
        import trimesh
        for label, verts in [("egoemg", ego_verts), ("manus", manus_verts)]:
            m = trimesh.Trimesh(vertices=verts, faces=faces[:, ::-1], process=False)
            m.export(str(out_dir / f"frame_{i:02d}_{label}.glb"))

    print(f"\nDone. Output in {out_dir}/")
    print(f"  frame_*_combined.glb: blue=EgoEMG(MANO), red=Manus(UmeTrack)")
    print(f"  frame_*_egoemg.glb:   EgoEMG angles only")
    print(f"  frame_*_manus.glb:    Manus angles only")


if __name__ == "__main__":
    main()
