#!/usr/bin/env python

# Utility: recompute IK masks from original (unmodified) HDF5 files and
# write them into a set of target files whose joint_angles may have been
# interpolated/modified. Only the mask datasets are touched.

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from emg2pose.data import Emg2PoseSessionData
from emg2pose.utils import get_ik_failures_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute IK masks from original HDF5 files and write them into "
            "target files that may have damaged masks. Only mask datasets are modified."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Root directory of original (clean) HDF5 files.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Root directory of current HDF5 files to repair.",
    )
    parser.add_argument(
        "--pattern",
        default="*.hdf5",
        help="Glob pattern (relative to src root) to select session files. Default: *.hdf5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be updated without writing changes.",
    )
    return parser.parse_args()


def compute_masks_from_source(src_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(src_path, "r") as f:
        if Emg2PoseSessionData.HDF5_GROUP not in f:
            raise KeyError(f"missing group {Emg2PoseSessionData.HDF5_GROUP} in {src_path}")
        group = f[Emg2PoseSessionData.HDF5_GROUP]
        ts = group[Emg2PoseSessionData.TIMESERIES]
        joint_angles = ts[Emg2PoseSessionData.JOINT_ANGLES]
        no_ik_mask = get_ik_failures_mask(joint_angles).astype(np.bool_)
        ik_failure_mask = ~no_ik_mask
        return no_ik_mask, ik_failure_mask, len(ts)


def write_masks(dst_path: Path, no_mask: np.ndarray, ik_mask: np.ndarray) -> None:
    with h5py.File(dst_path, "r+") as f:
        group = f[Emg2PoseSessionData.HDF5_GROUP]
        for name in ("no_ik_failure", "ik_failure_mask"):
            if name in group:
                del group[name]

        ds_no = group.create_dataset("no_ik_failure", data=no_mask)
        ds_no.attrs["description"] = (
            "Boolean mask where True indicates no IK failure; recomputed from original joint_angles."
        )
        ds_ik = group.create_dataset("ik_failure_mask", data=ik_mask)
        ds_ik.attrs["description"] = (
            "Boolean mask where True indicates original IK failure (zeros) before interpolation."
        )


def main() -> None:
    args = parse_args()
    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"src does not exist: {src_root}")
    if not dst_root.exists():
        raise FileNotFoundError(f"dst does not exist: {dst_root}")

    src_files = sorted(src_root.rglob(args.pattern))
    if not src_files:
        print(f"No source files found under {src_root} with pattern {args.pattern}")
        return

    counts = {"written": 0, "skipped": 0, "missing_dst": 0, "len_mismatch": 0, "missing_group": 0}

    for src_path in tqdm(src_files, desc="Restoring IK masks"):
        dst_path = dst_root / src_path.relative_to(src_root)
        if not dst_path.exists():
            counts["missing_dst"] += 1
            continue
        try:
            no_mask, ik_mask, src_len = compute_masks_from_source(src_path)
        except KeyError:
            counts["missing_group"] += 1
            continue

        with h5py.File(dst_path, "r") as f_dst:
            if Emg2PoseSessionData.HDF5_GROUP not in f_dst:
                counts["missing_group"] += 1
                continue
            dst_group = f_dst[Emg2PoseSessionData.HDF5_GROUP]
            dst_ts = dst_group[Emg2PoseSessionData.TIMESERIES]
            dst_len = len(dst_ts)

        if src_len != dst_len:
            counts["len_mismatch"] += 1
            continue

        if args.dry_run:
            counts["written"] += 1
            continue

        write_masks(dst_path, no_mask, ik_mask)
        counts["written"] += 1

    print(
        f"Done. written={counts['written']}, skipped={counts['skipped']}, "
        f"missing_dst={counts['missing_dst']}, len_mismatch={counts['len_mismatch']}, "
        f"missing_group={counts['missing_group']}"
    )


if __name__ == "__main__":
    main()
