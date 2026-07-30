#!/usr/bin/env python3
"""
Convert EMG2Pose zarr dataset to numpy memmap format (global-concatenated).

Produces flat memmap files that can be sliced with zero-copy random access,
eliminating the Zarr Python overhead (~98% of read latency).

Includes complete session metadata for split filtering and normalization stats.

Usage:
    # Full conversion (all fields):
    python scripts/data/convert_emg2pose_zarr_to_memmap.py \
        --zarr-root /path/to/emg2pose_v3 \
        --out-root /path/to/emg2pose_v3_memmap \
        --fields emg joint_angles valid_mask time

    # Incremental conversion (add missing fields):
    python scripts/data/convert_emg2pose_zarr_to_memmap.py \
        --zarr-root /path/to/emg2pose_v3 \
        --out-root /path/to/emg2pose_v3_memmap \
        --fields joint_angles time \
        --append

    # Small-scale test:
    python scripts/data/convert_emg2pose_zarr_to_memmap.py \
        --zarr-root /path/to/emg2pose_v3 \
        --out-root /path/to/emg2pose_v3_memmap_test \
        --fields emg joint_angles valid_mask \
        --max-rows 1000000
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


def decode_bytes(arr: np.ndarray) -> list[str]:
    """Decode byte strings to UTF-8."""
    result = []
    for v in arr:
        if isinstance(v, (bytes, np.bytes_)):
            result.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            result.append(str(v))
    return result


def convert_field(
    zarr_arr: zarr.Array,
    out_path: Path,
    dtype: np.dtype,
    total_rows: int,
    max_rows: int | None = None,
) -> None:
    """Convert a single zarr array to memmap."""
    if max_rows is not None:
        shape = (max_rows,) + zarr_arr.shape[1:]
    else:
        shape = zarr_arr.shape

    mm = np.memmap(str(out_path), dtype=dtype, mode="w+", shape=shape)

    rows_to_write = min(total_rows, max_rows or total_rows)
    written = 0

    with tqdm(total=rows_to_write, desc=out_path.name, unit="row", unit_scale=True) as pbar:
        while written < rows_to_write:
            end = min(written + CHUNK_ROWS, rows_to_write)
            chunk = np.asarray(zarr_arr[written:end])
            mm[written:end] = chunk.astype(dtype, copy=False)
            pbar.update(end - written)
            written = end

    mm.flush()
    del mm


def build_session_metadata(root: zarr.Group, max_rows: int | None = None) -> list[dict]:
    """Build session index for manifest."""
    sessions = root["sessions"]

    session_id = decode_bytes(np.asarray(sessions["session_id"]))
    filename = decode_bytes(np.asarray(sessions["filename"]))
    user_id = np.asarray(sessions["user_id"], dtype=np.int32)
    stage_id = np.asarray(sessions["stage_id"], dtype=np.int32)
    side_id = np.asarray(sessions["side_id"], dtype=np.int8)
    split_id = np.asarray(sessions["split_id"], dtype=np.int32)
    start_idx = np.asarray(sessions["start_idx"], dtype=np.int64)
    length = np.asarray(sessions["length"], dtype=np.int64)
    end_idx = np.asarray(sessions["end_idx"], dtype=np.int64)

    # Filter sessions by max_rows
    session_list = []
    for i in range(len(session_id)):
        sess_start = int(start_idx[i])
        sess_end = int(end_idx[i])

        if max_rows is not None and sess_start >= max_rows:
            continue

        # Clip session end if it exceeds max_rows
        if max_rows is not None:
            sess_end = min(sess_end, max_rows)
            sess_length = sess_end - sess_start
        else:
            sess_length = int(length[i])

        session_list.append({
            "session_id": session_id[i],
            "filename": filename[i],
            "user_id": int(user_id[i]),
            "stage_id": int(stage_id[i]),
            "side_id": int(side_id[i]),
            "split_id": int(split_id[i]),
            "start_idx": sess_start,
            "length": sess_length,
            "end_idx": sess_end,
        })

    return session_list


def build_other_metadata(root: zarr.Group) -> dict:
    """Build users, stages, sides, splits metadata."""
    users = decode_bytes(np.asarray(root["users"]["user"]))
    stages = decode_bytes(np.asarray(root["stages"]["stage"]))
    sides = decode_bytes(np.asarray(root["sides"]["side"]))
    splits = decode_bytes(np.asarray(root["splits"]["split"]))

    return {
        "users": users,
        "stages": stages,
        "sides": sides,
        "splits": splits,
    }


def build_stats_metadata(root: zarr.Group, max_rows: int | None = None) -> dict:
    """Build normalization stats for manifest."""
    stats = root["stats"]

    session_count = np.asarray(stats["session_count"], dtype=np.int64)
    session_sum = np.asarray(stats["session_sum"], dtype=np.float64)
    session_sumsq = np.asarray(stats["session_sumsq"], dtype=np.float64)

    # Convert to lists for JSON serialization
    return {
        "session_count": session_count.tolist(),
        "session_sum": session_sum.tolist(),
        "session_sumsq": session_sumsq.tolist(),
    }


def build_blocks_metadata(root: zarr.Group) -> dict:
    """Build blocks metadata for skip_ik_failures functionality."""
    if "blocks" not in root:
        return {}

    blocks = root["blocks"]

    session_idx = np.asarray(blocks["session_idx"], dtype=np.int32)
    start = np.asarray(blocks["start"], dtype=np.int64)
    end = np.asarray(blocks["end"], dtype=np.int64)
    length = np.asarray(blocks["length"], dtype=np.int32)

    return {
        "session_idx": session_idx.tolist(),
        "start": start.tolist(),
        "end": end.tolist(),
        "length": length.tolist(),
    }


def convert_emg2pose(
    zarr_root: Path,
    out_root: Path,
    fields: list[str],
    overwrite: bool,
    append: bool,
    max_rows: int | None = None,
) -> None:
    """Convert EMG2Pose zarr to memmap format."""
    if not zarr_root.exists():
        print(f"Error: {zarr_root} not found")
        return

    if out_root.exists() and not overwrite and not append:
        print(f"Skip: {out_root} exists (use --overwrite or --append)")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    # Open zarr store
    root = zarr.open_group(str(zarr_root), mode="r")

    # Get total rows from emg array
    emg = root["emg"]
    total_rows = emg.shape[0]
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)

    print(f"EMG2Pose: {total_rows:,} rows")

    # Load or update manifest
    manifest_path = out_root / "manifest.json"
    if append and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  Appending to existing manifest ({len(manifest['fields'])} fields)")
    else:
        manifest = {
            "total_rows": total_rows,
            "fields": {},
            "sessions": [],
            "users": [],
            "stages": [],
            "sides": [],
            "splits": [],
            "stats": {},
        }

    # Convert each field
    for field_name in fields:
        if field_name not in root:
            print(f"  Warning: {field_name} not found in zarr, skipping")
            continue

        zarr_arr = root[field_name]
        dtype = zarr_arr.dtype
        shape = zarr_arr.shape if max_rows is None else (max_rows,) + zarr_arr.shape[1:]
        out_path = out_root / f"{field_name}.dat"

        if out_path.exists() and not overwrite:
            print(f"  Skip {field_name}: {out_path} exists")
            continue

        print(f"\n  Converting {field_name}: shape={shape}, dtype={dtype}")
        size_gb = np.prod(shape) * dtype.itemsize / (1024**3)
        print(f"    Output size: {size_gb:.1f} GB")

        t0 = time.time()
        convert_field(zarr_arr, out_path, dtype, total_rows, max_rows)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s ({total_rows / elapsed / 1e6:.1f}M rows/s)")

        manifest["fields"][field_name] = {
            "filename": out_path.name,
            "dtype": str(dtype),
            "shape": list(shape),
        }

    # Add metadata if not already present or doing full conversion
    if not append or not manifest["sessions"]:
        print("\n  Building session metadata...")
        manifest["sessions"] = build_session_metadata(root, max_rows)
        print(f"    {len(manifest['sessions'])} sessions")

        print("  Building other metadata...")
        other_meta = build_other_metadata(root)
        manifest["users"] = other_meta["users"]
        manifest["stages"] = other_meta["stages"]
        manifest["sides"] = other_meta["sides"]
        manifest["splits"] = other_meta["splits"]

        print("  Building stats metadata...")
        manifest["stats"] = build_stats_metadata(root, max_rows)

        print("  Building blocks metadata...")
        manifest["blocks"] = build_blocks_metadata(root)
        if manifest["blocks"]:
            print(f"    {len(manifest['blocks']['session_idx'])} blocks")

    # Write manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written to {manifest_path}")
    print(f"    Total fields: {len(manifest['fields'])}")
    print(f"    Total sessions: {len(manifest['sessions'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EMG2Pose zarr dataset to memmap format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=Path("data/emg_corpus/emg2pose_v3"),
        help="Input zarr root directory.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/emg_corpus/emg2pose_v3_memmap"),
        help="Output memmap root directory.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["emg", "joint_angles", "valid_mask", "time"],
        help="Zarr array names to convert (default: emg joint_angles valid_mask time).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing memmap files.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append fields to existing memmap directory.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to convert (for testing).",
    )
    args = parser.parse_args()

    convert_emg2pose(
        args.zarr_root,
        args.out_root,
        args.fields,
        args.overwrite,
        args.append,
        args.max_rows,
    )
    print("\nConversion complete!")


if __name__ == "__main__":
    main()