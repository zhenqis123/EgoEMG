#!/usr/bin/env python3
"""Add split, optical-camera, and wrist-label fields to a ShowEE shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


SPLIT_NAMES = ("train", "val", "test")


def create_field(root: Path, manifest: dict, name: str, dtype: str, shape: tuple[int, ...]):
    filename = f"{name}.dat"
    manifest["fields"][name] = {
        "filename": filename,
        "dtype": dtype,
        "shape": list(shape),
    }
    return np.memmap(root / filename, mode="w+", dtype=dtype, shape=shape)


def decode(values: np.ndarray) -> np.ndarray:
    return np.char.decode(values, "utf-8")


def nearest_valid_indices(valid: np.ndarray) -> np.ndarray:
    good = np.flatnonzero(valid)
    if len(good) == 0:
        return np.zeros(len(valid), dtype=np.int64)
    positions = np.arange(len(valid))
    insertion = np.searchsorted(good, positions)
    left = good[np.maximum(insertion - 1, 0)]
    right = good[np.minimum(insertion, len(good) - 1)]
    return np.where(positions - left <= right - positions, left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.memmap_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata_path = root / "metadata.npz"
    metadata_file = np.load(metadata_path, allow_pickle=False)
    metadata = {key: metadata_file[key] for key in metadata_file.files}
    metadata_file.close()

    total = int(manifest["total_rows"])
    starts = metadata["episode_start_idx"].astype(np.int64)
    ends = metadata["episode_end_idx"].astype(np.int64)
    subjects = decode(metadata["episode_subject"])
    sources = decode(metadata["episode_source_parquet"])
    source_root = Path(manifest["source_root"])

    frame_split = create_field(root, manifest, "frame_split_id", "int32", (total,))
    camera_transform = create_field(
        root, manifest, "mocap_webcam_transform", "float32", (total, 12)
    )
    wrist_fields = {}
    for hand in ("left", "right"):
        wrist_fields[f"mocap_{hand}_wrist_pitch"] = create_field(
            root, manifest, f"mocap_{hand}_wrist_pitch", "float32", (total,)
        )
        wrist_fields[f"mocap_{hand}_wrist_yaw"] = create_field(
            root, manifest, f"mocap_{hand}_wrist_yaw", "float32", (total,)
        )
        wrist_fields[f"mocap_{hand}_wrist_angles_valid"] = create_field(
            root, manifest, f"mocap_{hand}_wrist_angles_valid", "bool", (total,)
        )

    pos_info = manifest["fields"]["mocap_webcam_position"]
    quat_info = manifest["fields"]["mocap_webcam_orientation"]
    webcam_position = np.memmap(
        root / pos_info["filename"], mode="r+", dtype=pos_info["dtype"], shape=tuple(pos_info["shape"])
    )
    webcam_orientation = np.memmap(
        root / quat_info["filename"], mode="r+", dtype=quat_info["dtype"], shape=tuple(quat_info["shape"])
    )

    episode_split_ids = np.empty(len(starts), dtype=np.int32)
    for episode, (start, end, subject, relative) in enumerate(
        zip(starts, ends, subjects, sources, strict=True)
    ):
        split_id = 0 if int(subject) <= 69 else (1 if int(subject) == 70 else 2)
        episode_split_ids[episode] = split_id
        frame_split[start:end] = split_id
        mocap_path = source_root / relative / "luster_mocap/mocap.h5"
        with h5py.File(mocap_path, "r") as handle:
            source_count = len(handle["timestamp"])
            source_indices = np.rint(
                np.linspace(0, source_count - 1, int(end - start))
            ).astype(np.int64)
            position = handle["head/cam_position"][:].astype(np.float32) / 1000.0
            quaternion = handle["head/cam_quaternion"][:].astype(np.float32)
            tracked = handle["head/is_tracked"][:].astype(bool)
            pose_valid = (
                tracked
                & np.isfinite(position).all(axis=1)
                & np.isfinite(quaternion).all(axis=1)
                & (np.linalg.norm(quaternion, axis=1) > 0.5)
            )
            fill_indices = nearest_valid_indices(pose_valid)
            position = position[fill_indices]
            quaternion = quaternion[fill_indices]
            rotation = Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
            camera_transform[start:end, :9] = rotation[source_indices].reshape(-1, 9)
            camera_transform[start:end, 9:] = position[source_indices]
            webcam_position[start:end] = position[source_indices]
            webcam_orientation[start:end] = quaternion[source_indices]
        if (episode + 1) % 50 == 0 or episode + 1 == len(starts):
            print(f"[{episode + 1}/{len(starts)}] {relative}", flush=True)

    for array in (frame_split, camera_transform, webcam_position, webcam_orientation, *wrist_fields.values()):
        array.flush()

    metadata["episode_split_id"] = episode_split_ids
    metadata["splits_split"] = np.asarray(SPLIT_NAMES, dtype="S5")
    metadata["splits_split_id"] = np.arange(len(SPLIT_NAMES), dtype=np.int32)
    np.savez(metadata_path, **metadata)
    manifest["showee_split_policy"] = {
        "train": ["0061", "0062", "0063", "0064", "0065", "0066", "0067", "0068", "0069"],
        "val": ["0070"],
        "test": ["0071", "0072"],
    }
    manifest["wrist_angle_policy"] = "zero values with valid=false (not derivable with verified EgoEMG convention)"
    manifest["mocap_webcam_pose_source"] = "head/cam_position + head/cam_quaternion (xyzw)"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done: {root}")


if __name__ == "__main__":
    main()
