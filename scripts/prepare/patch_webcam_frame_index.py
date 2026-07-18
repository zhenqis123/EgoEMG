"""Patch image_webcam_frame_index.dat using parquet ground truth.

The memmap's image_webcam_frame_index was built by making wf values contiguous,
which squeezed out the freeze gaps. This script restores the correct mapping
from the parquet files, replacing -1 (stale) values with the last valid frame.

Usage:
    python scripts/prepare/patch_webcam_frame_index.py --episodes episode_000005

Safety: creates a backup of the .dat file before modifying.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

MEMMAP_DIR = Path("/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap")
PARQUET_DIR = Path("/home/xiziheng/develop/emg2pose/data/EgoEMG/data/chunk-000")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load metadata
    md = dict(np.load(MEMMAP_DIR / "metadata.npz", allow_pickle=False))
    ep_ids = [x.decode("utf-8").rstrip("\x00") if isinstance(x, (bytes, np.bytes_)) else str(x)
              for x in md["episode_id"]]
    ep_starts = np.asarray(md["episode_start_idx"], dtype=np.int64)
    ep_ends = np.asarray(md["episode_end_idx"], dtype=np.int64)

    # Load manifest
    with open(MEMMAP_DIR / "manifest.json") as f:
        manifest = json.load(f)
    field_info = manifest["fields"]["image_webcam_frame_index"]
    dat_path = MEMMAP_DIR / field_info["filename"]
    dtype = np.dtype(field_info["dtype"])
    shape = tuple(field_info["shape"])

    # Backup
    backup_path = dat_path.with_suffix(".dat.bak")
    if not backup_path.exists() or args.dry_run:
        print(f"Creating backup: {backup_path}")
        if not args.dry_run:
            shutil.copy2(dat_path, backup_path)
    else:
        print(f"Backup already exists: {backup_path}")

    if args.episodes:
        selected = [(i, eid) for i, eid in enumerate(ep_ids) if eid in args.episodes]
    else:
        selected = list(enumerate(ep_ids))

    mm = np.memmap(dat_path, dtype=dtype, mode="r+" if not args.dry_run else "r",
                   shape=shape)

    total_patched = 0
    for ep_idx, ep_id in tqdm(selected, desc="Episodes", unit="ep"):
        start = int(ep_starts[ep_idx])
        end = int(ep_ends[ep_idx])
        n_rows = end - start

        # Read parquet ground truth
        pq_path = PARQUET_DIR / f"{ep_id}.parquet"
        df = pd.read_parquet(pq_path, columns=["observation.images.webcam.frame_index"],
                             engine="pyarrow")
        true_wf = df["observation.images.webcam.frame_index"].values.astype(np.int32)

        if len(true_wf) != n_rows:
            tqdm.write(f"  {ep_id}: row count mismatch (parquet={len(true_wf)}, memmap={n_rows}), skip")
            continue

        # Forward-fill -1 values
        last_valid = 0
        stale_count = 0
        for i in range(len(true_wf)):
            if true_wf[i] == -1:
                true_wf[i] = last_valid
                stale_count += 1
            else:
                last_valid = int(true_wf[i])

        if stale_count == 0:
            tqdm.write(f"  {ep_id}: no stale frames, skip")
            continue

        # Check how many values actually differ from current memmap
        cur_mm = np.array(mm[start:end])
        diffs = (cur_mm != true_wf).sum()
        tqdm.write(f"  {ep_id}: {stale_count} stale → forward-filled, "
                   f"{diffs}/{n_rows} values changed ({100*diffs/n_rows:.2f}%)")

        if not args.dry_run:
            mm[start:end] = true_wf
        total_patched += stale_count

    if args.dry_run:
        print(f"\nDRY RUN: would patch {total_patched} stale entries across {len(selected)} episodes")
    else:
        print(f"\nPatched {total_patched} stale entries across {len(selected)} episodes")
        print(f"Backup at: {backup_path}")


if __name__ == "__main__":
    main()
