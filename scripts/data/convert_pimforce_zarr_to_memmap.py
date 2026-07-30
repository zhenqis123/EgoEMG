#!/usr/bin/env python3
"""
Convert Pimforce zarr dataset to numpy memmap format (global-concatenated).

Produces flat memmap files that can be sliced with zero-copy random access,
eliminating the Zarr Python overhead (~98% of read latency).

Usage:
    python scripts/data/convert_pimforce_zarr_to_memmap.py \
        --zarr-root /path/to/pimforce_v3 \
        --out-root /path/to/pimforce_v3_memmap
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


def convert_pimforce(
    zarr_root: Path,
    out_root: Path,
    overwrite: bool,
) -> None:
    """Convert Pimforce zarr to memmap format."""
    if not zarr_root.exists():
        print(f"Skip: {zarr_root} not found")
        return

    if out_root.exists() and not overwrite:
        print(f"Skip: {out_root} exists (use --overwrite)")
        return
    out_root.mkdir(parents=True, exist_ok=True)

    # Open zarr store
    root = zarr.open_group(str(zarr_root), mode="r")

    # Read session metadata
    sessions = root["sessions"]
    session_name_arr = np.asarray(sessions["session_name"])
    user_id = np.asarray(sessions["user_id"], dtype=np.int32)
    start_idx = np.asarray(sessions["start_idx"], dtype=np.int64)
    length = np.asarray(sessions["length"], dtype=np.int64)

    print(f"Pimforce: {len(session_name_arr)} sessions")

    # Read data arrays
    emg = root["emg"]
    joint_angles = root["joint_angles"] if "joint_angles" in root else None
    force = root.get("force")
    valid_mask = root.get("valid_mask")

    total_rows = emg.shape[0]
    print(f"  Total rows: {total_rows:,}")

    # Fields to convert
    fields = [("emg", emg, np.float32)]
    if joint_angles is not None:
        fields.append(("joint_angles", joint_angles, np.float32))
    if force is not None:
        fields.append(("force", force, np.float32))
    if valid_mask is not None:
        fields.append(("valid_mask", valid_mask, np.bool_))

    manifest = {"total_rows": total_rows, "fields": {}, "sessions": []}

    # Build session index for memmap
    session_name_arr = np.asarray(sessions["session_name"])
    for i in range(len(session_name_arr)):
        sess_name = session_name_arr[i].decode("utf-8") if isinstance(session_name_arr[i], bytes) else str(session_name_arr[i])
        manifest["sessions"].append({
            "session_name": sess_name,
            "user_id": int(user_id[i]),
            "start_idx": int(start_idx[i]),
            "length": int(length[i]),
        })

    # Convert each field to memmap
    for field_name, zarr_arr, dtype in fields:
        shape = zarr_arr.shape
        dtype = np.dtype(dtype)  # Ensure dtype is a numpy dtype object
        out_path = out_root / f"{field_name}.dat"

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
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Pimforce zarr dataset to memmap format."
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
        default=Path("data/emg_corpus/pimforce_v3_memmap"),
        help="Output memmap root directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing memmap files.",
    )
    args = parser.parse_args()

    convert_pimforce(args.zarr_root, args.out_root, args.overwrite)
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
