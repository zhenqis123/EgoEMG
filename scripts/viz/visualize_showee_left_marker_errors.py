#!/usr/bin/env python3
"""Export high-error ShowEE left-marker frames as GT/prediction skeleton GLBs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh

from markers2mano.geometry import (
    matrix_to_axis_angle,
    transform_joints_coordinates_torch,
)
from markers2mano.rigid_align import compute_aligned_error_batched
from scripts.mano.infer_mano_for_egoemg import (
    MARKER_VERT_INDICES,
    load_mano_layer,
    load_model,
    mirror_local_points_z,
    six_d_to_rot_matrix,
)


BONES = tuple(
    edge
    for base in (1, 5, 9, 13, 17)
    for edge in ((0, base), (base, base + 1), (base + 1, base + 2), (base + 2, base + 3))
)


def add_skeleton(
    scene: trimesh.Scene,
    points: np.ndarray,
    color: tuple[int, int, int, int],
    prefix: str,
) -> None:
    for index, point in enumerate(points):
        marker = trimesh.creation.icosphere(subdivisions=2, radius=0.0035)
        marker.apply_translation(point)
        marker.visual.face_colors = color
        scene.add_geometry(marker, node_name=f"{prefix}_marker_{index:02d}")
    for start, end in BONES:
        bone = trimesh.creation.cylinder(
            radius=0.0016, segment=np.stack([points[start], points[end]])
        )
        bone.visual.face_colors = color
        scene.add_geometry(bone, node_name=f"{prefix}_bone_{start:02d}_{end:02d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showee-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-per-episode", type=int, default=2)
    parser.add_argument("--min-frame-gap", type=int, default=30)
    parser.add_argument("episodes", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    mano = load_mano_layer(device)
    marker_indices = MARKER_VERT_INDICES.to(device)
    candidates: list[dict[str, object]] = []

    for relative_path in args.episodes:
        mocap_path = args.showee_root / relative_path / "luster_mocap/mocap.h5"
        with h5py.File(mocap_path, "r") as handle:
            markers = handle["left_hand/markers"][:].astype(np.float32) / 1000.0
        valid_indices = np.flatnonzero(np.isfinite(markers).all(axis=(1, 2)))
        for offset in range(0, len(valid_indices), args.batch_size):
            indices = valid_indices[offset : offset + args.batch_size]
            world = torch.from_numpy(markers[indices]).to(device)
            centered = world - world[:, :1]
            local, _ = transform_joints_coordinates_torch(centered)
            local = mirror_local_points_z(local)
            with torch.no_grad():
                pose_6d, beta = model(local)
                pose = matrix_to_axis_angle(
                    six_d_to_rot_matrix(pose_6d.view(-1, 16, 6))
                ).view(-1, 48)
                pose[:, :3] = 0.0
                predicted = mano(pose, beta).verts[:, marker_indices]
                errors, rotations, translations = compute_aligned_error_batched(
                    predicted, local
                )
                aligned = torch.bmm(predicted, rotations.transpose(1, 2))
                aligned += translations[:, None]
            mean_errors = errors.mean(dim=1).mul(1000).cpu().numpy()
            local_np = local.cpu().numpy()
            aligned_np = aligned.cpu().numpy()
            for j, frame_index in enumerate(indices):
                candidates.append(
                    {
                        "episode": relative_path,
                        "frame_index": int(frame_index),
                        "error_mm": float(mean_errors[j]),
                        "gt": local_np[j],
                        "pred": aligned_np[j],
                    }
                )

    selected: list[dict[str, object]] = []
    selected_frames: dict[str, list[int]] = {}
    for item in sorted(candidates, key=lambda value: float(value["error_mm"]), reverse=True):
        episode = str(item["episode"])
        frame_index = int(item["frame_index"])
        previous = selected_frames.setdefault(episode, [])
        if len(previous) >= args.max_per_episode:
            continue
        if any(abs(frame_index - other) < args.min_frame_gap for other in previous):
            continue
        selected.append(item)
        previous.append(frame_index)
        if len(selected) == args.top_k:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for rank, item in enumerate(selected, 1):
        scene = trimesh.Scene()
        add_skeleton(scene, item["gt"], (40, 220, 100, 255), "gt")
        add_skeleton(scene, item["pred"], (230, 50, 190, 210), "prediction")
        stem = Path(str(item["episode"])).name
        filename = (
            f"rank_{rank:02d}_{stem}_frame_{int(item['frame_index']):06d}_"
            f"error_{float(item['error_mm']):.2f}mm.glb"
        )
        scene.export(args.output_dir / filename)
        report.append(
            {
                "rank": rank,
                "file": filename,
                "episode": item["episode"],
                "frame_index": item["frame_index"],
                "mean_aligned_marker_error_mm": item["error_mm"],
            }
        )
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "marker_shape_per_frame": [21, 3],
                "bones": BONES,
                "colors": {"ground_truth": "green", "prediction": "magenta"},
                "frames": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported {len(report)} GLBs to {args.output_dir}")


if __name__ == "__main__":
    main()
