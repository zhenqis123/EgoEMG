#!/usr/bin/env python3
"""Randomly sample raw ShowEE left-hand markers and export skeleton GLBs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import trimesh


BONES = tuple(
    edge
    for base in (1, 5, 9, 13, 17)
    for edge in ((0, base), (base, base + 1), (base + 1, base + 2), (base + 2, base + 3))
)


def add_skeleton(
    scene: trimesh.Scene,
    points: np.ndarray,
    hand: str,
) -> None:
    hand_color = (30, 190, 240, 255) if hand == "left" else (70, 220, 100, 255)
    wrist_color = (255, 90, 60, 255) if hand == "left" else (255, 210, 40, 255)
    for index, point in enumerate(points):
        marker = trimesh.creation.icosphere(subdivisions=2, radius=0.0035)
        marker.apply_translation(point)
        marker.visual.face_colors = wrist_color if index == 0 else hand_color
        scene.add_geometry(marker, node_name=f"{hand}_gt_marker_{index:02d}")
    for start, end in BONES:
        bone = trimesh.creation.cylinder(
            radius=0.0016, segment=np.stack([points[start], points[end]])
        )
        bone.visual.face_colors = hand_color
        scene.add_geometry(
            bone, node_name=f"{hand}_gt_bone_{start:02d}_{end:02d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showee-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--include-right", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mocap_paths = sorted(args.showee_root.glob("*/*/luster_mocap/mocap.h5"))
    if not mocap_paths:
        raise RuntimeError(f"No mocap files found below {args.showee_root}")

    lengths = []
    for path in mocap_paths:
        with h5py.File(path, "r") as handle:
            lengths.append(len(handle["left_hand/markers"]))
    cumulative = np.cumsum(np.asarray(lengths, dtype=np.int64))
    rng = np.random.default_rng(args.seed)
    selected: list[dict[str, object]] = []
    used: set[tuple[int, int]] = set()
    while len(selected) < args.num_frames:
        global_index = int(rng.integers(0, int(cumulative[-1])))
        episode_index = int(np.searchsorted(cumulative, global_index, side="right"))
        episode_start = 0 if episode_index == 0 else int(cumulative[episode_index - 1])
        frame_index = global_index - episode_start
        key = (episode_index, frame_index)
        if key in used:
            continue
        used.add(key)
        path = mocap_paths[episode_index]
        with h5py.File(path, "r") as handle:
            markers_mm = handle["left_hand/markers"][frame_index].astype(np.float32)
            right_markers_mm = handle["right_hand/markers"][frame_index].astype(
                np.float32
            )
        if not np.isfinite(markers_mm).all():
            continue
        selected.append(
            {
                "mocap_path": str(path.relative_to(args.showee_root)),
                "frame_index": frame_index,
                "markers_m": markers_mm / 1000.0,
                "right_markers_m": right_markers_mm / 1000.0,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for sample_index, item in enumerate(selected, 1):
        scene = trimesh.Scene()
        add_skeleton(scene, item["markers_m"], "left")
        right_valid = bool(np.isfinite(item["right_markers_m"]).all())
        if args.include_right and right_valid:
            add_skeleton(scene, item["right_markers_m"], "right")
        episode_name = Path(str(item["mocap_path"])).parts[1]
        filename = (
            f"random_{sample_index:02d}_{episode_name}_"
            f"frame_{int(item['frame_index']):06d}.glb"
        )
        scene.export(args.output_dir / filename)
        report.append(
            {
                "sample": sample_index,
                "file": filename,
                "mocap_path": item["mocap_path"],
                "frame_index": item["frame_index"],
                "right_hand_available": right_valid,
            }
        )
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "sampling": "uniform over all raw mocap frames; non-finite frames rejected",
                "hands": "left+right" if args.include_right else "left",
                "coordinates": "raw world coordinates converted from mm to m only",
                "marker_shape_per_frame": [21, 3],
                "bones": BONES,
                "colors": {
                    "left_wrist": "orange",
                    "left_gt": "cyan",
                    "right_wrist": "yellow",
                    "right_gt": "green",
                },
                "frames": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported {len(report)} raw GT skeletons to {args.output_dir}")


if __name__ == "__main__":
    main()
