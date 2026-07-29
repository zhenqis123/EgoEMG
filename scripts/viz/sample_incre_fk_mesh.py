"""Sample FK mesh GLBs from incre dataset episode 3 (val split).

Usage:
    PYOPENGL_PLATFORM=osmesa python scripts/viz/sample_incre_fk_mesh.py \
        --n_samples 10 \
        --output /tmp/incre_fk_samples
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util
_project_root = Path(__file__).resolve().parent.parent.parent
_viz_spec = importlib.util.spec_from_file_location(
    "emg2pose_viz_mod",
    str(_project_root / "emg2pose" / "visualization.py"),
)
_viz_mod = importlib.util.module_from_spec(_viz_spec)
_viz_spec.loader.exec_module(_viz_mod)
skin_mesh_from_angles = _viz_mod.skin_mesh_from_angles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memmap_dir",
        default="./data/EgoEMG_incre/data_right_merged",
    )
    parser.add_argument("--episode", type=int, default=3,
                        help="Episode index to sample from (default: 3, the last one)")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test"],
                        help="Split within the episode to sample from")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/incre_fk_mesh_samples")
    args = parser.parse_args()

    mm_dir = Path(args.memmap_dir)
    with open(mm_dir / "manifest.json") as f:
        manifest = json.load(f)

    # Load memmaps
    ja_mm = np.memmap(
        mm_dir / "generated_joint_angles_right.dat",
        dtype=np.float32,
        mode="r",
        shape=tuple(manifest["fields"]["generated_joint_angles_right"]["shape"]),
    )
    ep_mm = np.memmap(
        mm_dir / "episode_index.dat",
        dtype=np.int64,
        mode="r",
        shape=tuple(manifest["fields"]["episode_index"]["shape"]),
    )
    split_mm = np.memmap(
        mm_dir / "frame_split_id.dat",
        dtype=np.int8,
        mode="r",
        shape=tuple(manifest["fields"]["frame_split_id"]["shape"]),
    )

    # Find candidate frames: episode 3, specific split
    split_id = {"train": 0, "val": 1, "test": 2}[args.split]
    candidates = np.where((ep_mm == args.episode) & (split_mm == split_id))[0]
    print(f"Episode {args.episode} split='{args.split}': {len(candidates):,} candidate frames")

    if len(candidates) == 0:
        print("No frames found!")
        return

    rng = np.random.RandomState(args.seed)
    n = min(args.n_samples, len(candidates))
    sampled = sorted(rng.choice(candidates, size=n, replace=False))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # FK mesh via UmeTrack
    for i, global_idx in enumerate(sampled):
        ja = np.asarray(ja_mm[global_idx], dtype=np.float32)

        if not np.isfinite(ja).all() or np.abs(ja).sum() < 1e-6:
            print(f"[{i+1}/{n}] global_idx={global_idx}: SKIP (invalid angles)")
            continue

        try:
            # UmeTrack FK: always skin as right hand (incre is right-hand only)
            verts_local, faces = skin_mesh_from_angles(
                joint_angles=ja[:20], flip=False,
            )
            # Scale to reasonable hand size (same as visualize_egoemg_mesh.py)
            verts_local = verts_local.copy()
            fk_span = np.median(verts_local.max(axis=0) - verts_local.min(axis=0))
            if fk_span > 1e-6:
                verts_local = verts_local * (0.09 / fk_span)

            path = out_dir / f"fk_ep{args.episode}_{args.split}_{global_idx:08d}.glb"
            mesh = trimesh.Trimesh(vertices=verts_local, faces=faces, process=False)
            mesh.visual.vertex_colors = [255, 180, 0, 255]  # Orange
            mesh.export(str(path))
            print(f"[{i+1}/{n}] global_idx={global_idx} → {path.name}")

        except Exception as e:
            print(f"[{i+1}/{n}] global_idx={global_idx}: ERROR {e}")

    print(f"Done. Saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
