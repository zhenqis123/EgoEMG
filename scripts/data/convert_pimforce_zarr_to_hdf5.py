#!/usr/bin/env python3
"""
Convert Pimforce zarr dataset to HDF5 format (emg2pose format).

The zarr data already has joint_angles. We just need to copy them to HDF5 format.

Usage:
    python scripts/data/convert_pimforce_zarr_to_hdf5.py \
        --zarr-root /path/to/pimforce_v3 \
        --out-root /path/to/pimforce_v3_hdf5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import zarr
from tqdm import tqdm


def _convert_pimforce(
    zarr_root: Path,
    out_root: Path,
    overwrite: bool,
) -> None:
    """Convert Pimforce zarr to HDF5 format."""
    if not zarr_root.exists():
        print(f"Skip: {zarr_root} not found")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    # Open zarr store
    root = zarr.open_group(str(zarr_root), mode="r")

    # Read data arrays
    emg = root["emg"]
    joint_angles = root["joint_angles"] if "joint_angles" in root else None
    force = root.get("force")
    valid_mask = root.get("valid_mask")

    # Read session metadata
    sessions = root["sessions"]
    session_name_arr = np.asarray(sessions["session_name"])
    user_id = np.asarray(sessions["user_id"], dtype=np.int32)
    start_idx = np.asarray(sessions["start_idx"], dtype=np.int64)
    length = np.asarray(sessions["length"], dtype=np.int64)

    print(f"Pimforce: {len(session_name_arr)} sessions")
    print(f"  EMG shape: {emg.shape}")
    if joint_angles is not None:
        print(f"  Joint angles shape: {joint_angles.shape}")

    # Convert each session to HDF5
    converted = 0
    skipped = 0
    for i in tqdm(range(len(session_name_arr)), desc="Pimforce sessions"):
        sess_start = start_idx[i]
        sess_len = length[i]
        sess_end = sess_start + sess_len

        # Skip very short sessions
        if sess_len < 1000:
            skipped += 1
            continue

        # Read session data
        sess_emg = np.asarray(emg[sess_start:sess_end]).astype(np.float32)
        sess_joint_angles = None
        if joint_angles is not None:
            sess_joint_angles = np.asarray(joint_angles[sess_start:sess_end]).astype(np.float32)

        # Decode session name
        sess_name = session_name_arr[i].decode("utf-8") if isinstance(session_name_arr[i], bytes) else str(session_name_arr[i])
        user = str(user_id[i])

        # Write HDF5
        out_path = out_root / f"{sess_name}.hdf5"
        if out_path.exists() and not overwrite:
            continue

        if out_path.exists():
            out_path.unlink()

        attrs = {
            "dataset": "Pimforce",
            "session": sess_name,
            "user": user,
            "num_channels": int(sess_emg.shape[1]),
        }
        if sess_joint_angles is not None:
            attrs["num_joint_angles"] = int(sess_joint_angles.shape[1])

        with h5py.File(out_path, "w") as f:
            g = f.create_group("emg2pose")
            chunk_len = min(4096, int(sess_emg.shape[0])) or 1
            g.create_dataset("emg", data=sess_emg, chunks=(chunk_len, sess_emg.shape[1]))
            if sess_joint_angles is not None:
                # Convert degrees to radians
                g.create_dataset(
                    "joint_angles",
                    data=np.deg2rad(sess_joint_angles),
                    chunks=(chunk_len, sess_joint_angles.shape[1])
                )
            if force is not None:
                sess_force = np.asarray(force[sess_start:sess_end]).astype(np.float32)
                g.create_dataset("force", data=sess_force)
            if valid_mask is not None:
                sess_valid = np.asarray(valid_mask[sess_start:sess_end]).astype(np.float32)
                g.create_dataset("valid_mask", data=sess_valid)
            g.attrs.update(attrs)

        converted += 1

    print(f"  Converted: {converted}, Skipped: {skipped}")
    print(f"  Output: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Pimforce zarr dataset to HDF5 format."
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path("data/emg_corpus/pimforce_v3"),
        help="Input zarr root directory.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/emg_corpus/pimforce_v3_hdf5"),
        help="Output HDF5 root directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 files.",
    )
    args = parser.parse_args()

    _convert_pimforce(args.zarr_root, args.out_root, args.overwrite)
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
