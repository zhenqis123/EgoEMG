#!/usr/bin/env python3
"""Export raw bimanual marker skeletons with aligned MANO meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh

from markers2mano.geometry import matrix_to_axis_angle, transform_joints_coordinates_torch
from markers2mano.rigid_align import compute_aligned_error
from scripts.mano.infer_mano_for_egoemg import (
    MARKER_VERT_INDICES,
    load_mano_layer,
    load_model,
    mirror_local_points_z,
    six_d_to_rot_matrix,
)
from scripts.viz.export_showee_random_gt_marker_skeletons import add_skeleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mocap", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    mano = load_mano_layer(device)
    faces = mano.th_faces.detach().cpu().numpy()
    marker_indices = MARKER_VERT_INDICES.to(device)
    scene = trimesh.Scene()
    report: dict[str, object] = {"frame": args.frame, "hands": {}}

    with h5py.File(args.mocap, "r") as handle:
        for hand in ("left", "right"):
            world = torch.from_numpy(
                handle[f"{hand}_hand/markers"][args.frame].astype(np.float32)
                / 1000.0
            ).to(device)
            if not torch.isfinite(world).all():
                report["hands"][hand] = {"available": False}
                continue
            add_skeleton(scene, world.cpu().numpy(), hand)

            centered = world[None] - world[None, :1]
            local, _ = transform_joints_coordinates_torch(centered)
            if hand == "left":
                local = mirror_local_points_z(local)
            with torch.no_grad():
                pose_6d, beta = model(local)
                pose = matrix_to_axis_angle(
                    six_d_to_rot_matrix(pose_6d.view(-1, 16, 6))
                ).view(-1, 48)
                pose[:, :3] = 0.0
                vertices = mano(pose, beta).verts[0]

            # Convert the canonical MANO-right left prediction back to a
            # left-hand geometry before aligning it to world-space GT markers.
            if hand == "left":
                vertices = mirror_local_points_z(vertices)
            predicted_markers = vertices[marker_indices]
            errors, rotation, translation = compute_aligned_error(
                predicted_markers, world
            )
            vertices_world = vertices @ rotation.T + translation
            mesh_faces = faces[:, [0, 2, 1]] if hand == "left" else faces
            mesh = trimesh.Trimesh(
                vertices=vertices_world.cpu().numpy(),
                faces=mesh_faces,
                process=False,
            )
            mesh.visual.face_colors = (
                (30, 120, 245, 125) if hand == "left" else (50, 210, 90, 125)
            )
            scene.add_geometry(mesh, node_name=f"{hand}_mano_mesh")
            error_mm = errors.mul(1000)
            report["hands"][hand] = {
                "available": True,
                "mean_aligned_marker_error_mm": float(error_mm.mean()),
                "median_aligned_marker_error_mm": float(error_mm.median()),
                "max_aligned_marker_error_mm": float(error_mm.max()),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output)
    report["output"] = args.output.name
    report["colors"] = {
        "left_skeleton": "cyan",
        "left_mesh": "transparent blue",
        "right_skeleton": "green",
        "right_mesh": "transparent green",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
