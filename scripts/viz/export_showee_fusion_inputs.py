#!/usr/bin/env python3
"""Export actual ShowEE fusion image inputs and their UmeTrack GT meshes.

The image tensor is produced by ``EgoEmgMemmapDataset`` using the same
per-episode crop store and ImageNet normalization as the full-data fusion
experiments.  Each sample directory contains the normalized tensor consumed by
the model, a losslessly de-normalized PNG for inspection, and a GLB made from
the corresponding generated UmeTrack angles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from egoemg.realtime_local.mesh_visualizer import UmeTrackMeshForwarder


IMAGENET_MEAN = 255.0 * np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = 255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memmap-dir", type=Path, default=Path("data/ShowEE_202607_memmap")
    )
    parser.add_argument(
        "--crops-dir", type=Path, default=Path("data/ShowEE_202607_crops")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("viz_results/showee_fusion_input_check_20260722"),
    )
    parser.add_argument("--num-per-hand", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def make_dataset(args: argparse.Namespace, hand: str) -> EgoEmgMemmapDataset:
    # ``skip_emg_loading`` selects the vision-only fast path, but the crop read
    # and normalization are identical to center-supervised fusion's image path.
    return EgoEmgMemmapDataset(
        memmap_dir=args.memmap_dir,
        window_length=12_000,
        stride=1_200,
        allowed_splits=["all"],
        modalities=["joint_angles", "labels"],
        target_hand=hand,
        emg_field_preference="filtered_paper",
        emg_layout="target_hand",
        dataset_name="showee",
        jitter=False,
        vision_patch_size=256,
        vision_num_frames=1,
        vision_frame_selection="center",
        per_episode_crops_dir=str(args.crops_dir),
        skip_emg_loading=True,
        center_target_only=True,
    )


def denormalize(chw: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(np.asarray(chw, dtype=np.float32), 0, -1)
    rgb = rgb * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def export_mesh(
    forwarder: UmeTrackMeshForwarder, angles: np.ndarray, hand: str, output: Path
) -> None:
    mesh = forwarder(angles)
    vertices = mesh.vertices.copy()
    faces = mesh.triangles.copy()
    # Generated ShowEE labels use the project's canonical right-hand semantic.
    # Reflect left only for an intuitive left/right display in a generic viewer.
    if hand == "left":
        vertices[:, 0] *= -1
        faces = faces[:, [0, 2, 1]]
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tri.visual.face_colors = (
        (40, 150, 245, 255) if hand == "left" else (60, 220, 100, 255)
    )
    trimesh.Scene([tri]).export(output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    forwarder = UmeTrackMeshForwarder()
    records: list[dict[str, object]] = []

    for hand in ("left", "right"):
        dataset = make_dataset(args, hand)
        candidates = rng.permutation(len(dataset))
        exported = 0
        used_episodes: set[int] = set()
        for dataset_idx in candidates:
            ep_idx, center_idx = dataset._resolve_index_to_center(int(dataset_idx))
            if ep_idx in used_episodes:
                continue
            sample = dataset[int(dataset_idx)]
            if not bool(np.asarray(sample["label_valid_mask"]).all()):
                continue
            vision = np.asarray(sample["vision_img"], dtype=np.float32)
            if vision.shape != (3, 256, 256) or not np.isfinite(vision).all():
                continue
            raw = denormalize(vision)
            if not raw.any():
                continue

            sample_dir = args.output_dir / f"{hand}_{exported + 1:02d}"
            sample_dir.mkdir(exist_ok=True)
            np.save(sample_dir / "model_input_normalized_chw.npy", vision)
            cv2.imwrite(
                str(sample_dir / "model_input.png"), cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            )
            angles = np.asarray(sample["joint_angles"], dtype=np.float32).reshape(-1)
            np.save(sample_dir / "umetrack_angles_22d.npy", angles)
            export_mesh(forwarder, angles, hand, sample_dir / "umetrack_gt.glb")

            record = {
                "hand": hand,
                "dataset_index": int(dataset_idx),
                "episode_index": ep_idx,
                "episode_id": str(dataset._episode_id[ep_idx]),
                "subject": str(dataset._episode_subject[ep_idx]),
                "center_global_frame": center_idx,
                "video_frame_index": int(np.asarray(sample["vision_frame_indices"])[0]),
                "files": {
                    "model_input_png": "model_input.png",
                    "model_input_tensor": "model_input_normalized_chw.npy",
                    "umetrack_angles": "umetrack_angles_22d.npy",
                    "umetrack_glb": "umetrack_gt.glb",
                },
            }
            (sample_dir / "metadata.json").write_text(json.dumps(record, indent=2))
            records.append({"directory": sample_dir.name, **record})
            used_episodes.add(ep_idx)
            exported += 1
            if exported == args.num_per_hand:
                break
        if exported != args.num_per_hand:
            raise RuntimeError(f"Only exported {exported}/{args.num_per_hand} {hand} samples")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "selection": "random valid sample from distinct ShowEE episodes per hand",
                "image_pipeline": "per-episode crop -> CHW -> ImageNet normalization",
                "fusion_window_length": 12000,
                "seed": args.seed,
                "samples": records,
            },
            indent=2,
        )
    )
    print(f"Exported {len(records)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
