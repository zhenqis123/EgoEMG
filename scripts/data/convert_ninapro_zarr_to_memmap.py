#!/usr/bin/env python3
"""
Convert Ninapro zarr datasets to numpy memmap format (global-concatenated).

Produces flat memmap files that can be sliced with zero-copy random access,
eliminating the Zarr Python overhead (~98% of read latency).

Usage:
    python scripts/data/convert_ninapro_zarr_to_memmap.py \
        --zarr-root /path/to/Ninapro_relabeled_zarr \
        --out-root /path/to/Ninapro_relabeled_memmap \
        --dbs DB1 DB2 DB5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm


CHUNK_ROWS = 500_000  # rows per read/write pass


def convert_db(
    db: str,
    zarr_root: Path,
    out_root: Path,
    overwrite: bool,
) -> None:
    """Convert a single Ninapro DB to memmap format."""
    db_zarr = zarr_root / db
    if not db_zarr.exists():
        print(f"Skip {db}: {db_zarr} not found")
        return

    out_db = out_root / db
    if out_db.exists() and not overwrite:
        print(f"Skip {db}: {out_db} exists (use --overwrite)")
        return
    out_db.mkdir(parents=True, exist_ok=True)

    # Open zarr store
    root = zarr.open_group(str(db_zarr), mode="r")

    # Read session metadata
    sessions = root["sessions"]
    session_id = np.asarray(sessions["session_id"])
    user = np.asarray(sessions["user"], dtype=np.int32)
    exercise = np.asarray(sessions["exercise"], dtype=np.int16)
    start_idx = np.asarray(sessions["start_idx"], dtype=np.int64)
    length = np.asarray(sessions["length"], dtype=np.int64)

    print(f"\n{db}: {len(session_id)} sessions")

    # Read data arrays
    emg = root["emg"]
    joint_angles = root["joint_angles"] if "joint_angles" in root else None
    gesture_id = root["gesture_id"] if "gesture_id" in root else None
    valid_mask = root["valid_mask"] if "valid_mask" in root else None

    total_rows = emg.shape[0]
    print(f"  Total rows: {total_rows:,}")

    # Fields to convert
    fields = [("emg", emg, np.float32)]
    if joint_angles is not None:
        fields.append(("joint_angles", joint_angles, np.float32))
    if gesture_id is not None:
        fields.append(("gesture_id", gesture_id, np.int32))
    if valid_mask is not None:
        fields.append(("valid_mask", valid_mask, np.bool_))

    manifest = {"total_rows": total_rows, "fields": {}, "sessions": []}

    # Build session index for memmap
    for i in range(len(session_id)):
        sess_name = session_id[i].decode("utf-8") if isinstance(session_id[i], bytes) else str(session_id[i])
        manifest["sessions"].append({
            "session_id": sess_name,
            "user": int(user[i]),
            "exercise": int(exercise[i]),
            "start_idx": int(start_idx[i]),
            "length": int(length[i]),
        })

    # Convert each field to memmap
    for field_name, zarr_arr, dtype in fields:
        shape = zarr_arr.shape
        dtype = np.dtype(dtype)  # Ensure dtype is a numpy dtype object
        out_path = out_db / f"{field_name}.dat"

        print(f"\n  Converting {field_name}: shape={shape}, dtype={dtype}")
        size_gb = np.prod(shape) * dtype.itemsize / (1024**3)
        print(f"    Output size: {size_gb:.1f} GB")

        t0 = time.time()
        written = 0
        mm = np.memmap(str(out_path), dtype=dtype, mode="w+", shape=shape)

        with tqdm(total=total_rows, desc=f"    {field_name}", unit="row", unit_scale=True) as pbar:
            while written < total_rows:
                end = min(written + CHUNK_ROWS, total_rows)
                chunk = np.asarray(zarr_arr[written:end])
                mm[written:end] = chunk.astype(dtype, copy=False)
                pbar.update(end - written)
                written = end

        mm.flush()
        del mm
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s ({total_rows / elapsed / 1e6:.1f}M rows/s)")

        manifest["fields"][field_name] = {
            "filename": out_path.name,
            "dtype": str(dtype),
            "shape": list(shape),
        }

    # Write manifest
    manifest_path = out_db / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Ninapro zarr datasets to memmap format."
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
        default=Path("data/emg_corpus/Ninapro_relabeled_memmap"),
        help="Output memmap root directory.",
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
        help="Overwrite existing memmap files.",
    )
    args = parser.parse_args()

    for db in args.dbs:
        convert_db(db, args.zarr_root, args.out_root, args.overwrite)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
