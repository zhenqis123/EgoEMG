#!/usr/bin/env python3
"""
Convert Ninapro zarr datasets to HDF5 format (emg2pose format).

The zarr data already has joint_angles in 22 dimensions (in degrees).
We just need to copy them to HDF5 format for faster loading.

Usage:
    python scripts/data/convert_ninapro_zarr_to_hdf5.py \
        --zarr-root /path/to/Ninapro_relabeled_zarr \
        --out-root /path/to/Ninapro_relabeled_hdf5 \
        --dbs DB1 DB2 DB5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import zarr
from tqdm import tqdm


def _convert_db(
    db: str,
    zarr_root: Path,
    out_root: Path,
    overwrite: bool,
) -> None:
    """Convert a single Ninapro DB to HDF5 format."""
    db_zarr = zarr_root / db
    if not db_zarr.exists():
        print(f"Skip {db}: {db_zarr} not found")
        return

    out_db = out_root / db
    out_db.mkdir(parents=True, exist_ok=True)

    # Open zarr store
    root = zarr.open_group(str(db_zarr), mode="r")

    # Read metadata
    emg = root["emg"]
    joint_angles = root["joint_angles"] if "joint_angles" in root else None
    gesture_id = root["gesture_id"] if "gesture_id" in root else None
    valid_mask = root["valid_mask"] if "valid_mask" in root else None

    # Read session metadata
    sessions = root["sessions"]
    session_id = np.asarray(sessions["session_id"])
    user = np.asarray(sessions["user"], dtype=np.int32)
    exercise = np.asarray(sessions["exercise"], dtype=np.int16)
    start_idx = np.asarray(sessions["start_idx"], dtype=np.int64)
    length = np.asarray(sessions["length"], dtype=np.int64)

    print(f"\n{db}: {len(session_id)} sessions")
    print(f"  EMG shape: {emg.shape}")
    if joint_angles is not None:
        print(f"  Joint angles shape: {joint_angles.shape}")

    # Convert each session to HDF5
    converted = 0
    skipped = 0
    for i in tqdm(range(len(session_id)), desc=f"{db} sessions"):
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
        sess_name = session_id[i].decode("utf-8") if isinstance(session_id[i], bytes) else str(session_id[i])
        user_id = str(user[i])
        ex_id = str(exercise[i])

        # Write HDF5
        out_path = out_db / f"{sess_name}.hdf5"
        if out_path.exists() and not overwrite:
            continue

        if out_path.exists():
            out_path.unlink()

        attrs = {
            "dataset": f"Ninapro_{db}",
            "session": sess_name,
            "user": user_id,
            "exercise": ex_id,
            "num_channels": int(sess_emg.shape[1]),
        }
        if sess_joint_angles is not None:
            attrs["num_joint_angles"] = int(sess_joint_angles.shape[1])

        with h5py.File(out_path, "w") as f:
            g = f.create_group("emg2pose")
            chunk_len = min(4096, int(sess_emg.shape[0])) or 1
            g.create_dataset("emg", data=sess_emg, chunks=(chunk_len, sess_emg.shape[1]))
            if sess_joint_angles is not None:
                # Note: angles are stored in degrees in zarr, convert to radians
                g.create_dataset(
                    "joint_angles",
                    data=np.deg2rad(sess_joint_angles),
                    chunks=(chunk_len, sess_joint_angles.shape[1])
                )
            if gesture_id is not None:
                sess_gesture = np.asarray(gesture_id[sess_start:sess_end]).astype(np.int32)
                g.create_dataset("gesture_id", data=sess_gesture)
            if valid_mask is not None:
                sess_valid = np.asarray(valid_mask[sess_start:sess_end]).astype(np.float32)
                g.create_dataset("valid_mask", data=sess_valid)
            g.attrs.update(attrs)

        converted += 1

    print(f"  Converted: {converted}, Skipped: {skipped}")
    print(f"  Output: {out_db}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Ninapro zarr datasets to HDF5 format."
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path("data/emg_corpus/Ninapro_relabeled_zarr"),
        help="Input zarr root directory.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/emg_corpus/Ninapro_relabeled_hdf5"),
        help="Output HDF5 root directory.",
    )
    parser.add_argument(
        "--dbs",
        nargs="+",
        default=["DB1", "DB2", "DB5"],
        help="Ninapro DBs to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 files.",
    )
    args = parser.parse_args()

    for db in args.dbs:
        _convert_db(db, args.zarr_root, args.out_root, args.overwrite)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
