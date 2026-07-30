"""
Convert a global-concatenated Zarr EMG corpus to numpy memmap files.

Produces flat memmap files that can be sliced with zero-copy random access,
eliminating the Zarr Python shard overhead (~98% of read latency).

Usage:
    # emg2pose (emg + valid_mask only for recon pretraining):
    python scripts/convert_data/zarr_to_memmap.py \
        --zarr-root data/emg_corpus/emg2pose_v3 \
        --output-dir data/emg_corpus/emg2pose_v3_memmap \
        --fields emg valid_mask

    # emg2qwerty (emg_left + emg_right + time):
    python scripts/convert_data/zarr_to_memmap.py \
        --zarr-root data/emg_corpus/emg2qwerty_v3 \
        --output-dir data/emg_corpus/emg2qwerty_v3_memmap \
        --fields emg_left emg_right time
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import zarr
except ImportError:
    raise SystemExit("zarr is required: pip install zarr")


CHUNK_ROWS = 500_000  # rows per read/write pass


def convert_field(
    zarr_arr: zarr.Array,
    out_path: Path,
    dtype: np.dtype,
    total_rows: int,
) -> None:
    shape = zarr_arr.shape
    mm = np.memmap(str(out_path), dtype=dtype, mode="w+", shape=shape)

    written = 0
    with tqdm(total=total_rows, desc=out_path.name, unit="row", unit_scale=True) as pbar:
        while written < total_rows:
            end = min(written + CHUNK_ROWS, total_rows)
            chunk = np.asarray(zarr_arr[written:end])
            mm[written:end] = chunk.astype(dtype, copy=False)
            pbar.update(end - written)
            written = end

    mm.flush()
    del mm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr-root", type=Path, required=True,
                        help="Path to the Zarr store root.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for memmap files.")
    parser.add_argument("--fields", nargs="+", required=True,
                        help="Zarr array names to convert (e.g. emg valid_mask emg_left emg_right time).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    zarr_root = args.zarr_root
    output_dir = args.output_dir

    if output_dir.exists() and not args.overwrite:
        print(f"Output dir exists: {output_dir}. Use --overwrite to replace.")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(zarr_root), mode="r")

    # Auto-detect total rows from first field
    first_arr = root[args.fields[0]]
    total_rows = first_arr.shape[0]
    print(f"Total rows: {total_rows:,}")

    manifest: dict = {"total_rows": total_rows, "fields": {}}

    for field_name in args.fields:
        zarr_arr = root[field_name]
        dtype = zarr_arr.dtype
        shape = zarr_arr.shape
        out_path = output_dir / f"{field_name}.dat"

        print(f"\nConverting {field_name}: shape={shape}, dtype={dtype}")
        size_gb = np.prod(shape) * dtype.itemsize / (1024**3)
        print(f"  Output size: {size_gb:.1f} GB")

        t0 = time.time()
        convert_field(zarr_arr, out_path, dtype, total_rows)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({total_rows / elapsed / 1e6:.1f}M rows/s)")

        manifest["fields"][field_name] = {
            "filename": out_path.name,
            "dtype": str(dtype),
            "shape": list(shape),
        }

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
