#!/usr/bin/env python

# Summarize IK failure/interpolation statistics across a dataset of HDF5 sessions.

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
            "Summarize IK failure and interpolation stats over HDF5 sessions. "
            "Relies on precomputed masks when available."
        )
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing emg2pose HDF5 session files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.hdf5",
        help="Glob pattern (relative to data_root) to select session files. Default: *.hdf5",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print per-file statistics in addition to the aggregated summary.",
    )
    parser.add_argument(
        "--recompute-if-missing",
        action="store_true",
        help="If masks are missing, recompute IK failure mask from joint_angles (may be inaccurate after interpolation).",
    )
    return parser.parse_args()


def load_masks(
    group: h5py.Group,
    joint_angles: np.ndarray,
    expected_len: int,
    allow_recompute: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return (ik_failure_mask, interpolated_mask). ik_failure_mask may be None."""
    ik_mask: np.ndarray | None = None
    if "ik_failure_mask" in group and len(group["ik_failure_mask"]) == expected_len:
        ik_mask = group["ik_failure_mask"][...].astype(bool)
    elif "no_ik_failure" in group and len(group["no_ik_failure"]) == expected_len:
        ik_mask = ~group["no_ik_failure"][...].astype(bool)
    elif allow_recompute:
        ik_mask = ~get_ik_failures_mask(joint_angles).astype(bool)

    interp_mask = np.zeros(expected_len, dtype=bool)
    if "interpolated_mask" in group and len(group["interpolated_mask"]) == expected_len:
        interp_mask = group["interpolated_mask"][...].astype(bool)
    return ik_mask, interp_mask


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    session_paths = sorted(data_root.rglob(args.pattern))
    if not session_paths:
        print(f"No session files found under {data_root} with pattern {args.pattern}")
        return

    agg = {
        "files_total": 0,
        "files_missing_mask": 0,
        "total_samples": 0,
        "orig_fail": 0,
        "interp": 0,
    }

    for session_path in tqdm(session_paths, desc="Summarizing IK stats"):
        with h5py.File(session_path, "r") as f:
            if Emg2PoseSessionData.HDF5_GROUP not in f:
                agg["files_missing_mask"] += 1
                continue
            group = f[Emg2PoseSessionData.HDF5_GROUP]
            ts = group[Emg2PoseSessionData.TIMESERIES]
            expected_len = len(ts)
            joint_angles = ts[Emg2PoseSessionData.JOINT_ANGLES]

            ik_mask, interp_mask = load_masks(
                group, joint_angles, expected_len, args.recompute_if_missing
            )
            if ik_mask is None:
                agg["files_missing_mask"] += 1
                continue

            orig_fail = int(np.sum(ik_mask))
            interp = int(np.sum(interp_mask))
            total = expected_len

            agg["files_total"] += 1
            agg["total_samples"] += total
            agg["orig_fail"] += orig_fail
            agg["interp"] += interp

            if args.per_file:
                remaining = max(orig_fail - interp, 0)
                print(
                    f"{session_path}: total={total}, "
                    f"orig_fail={orig_fail} ({orig_fail/total:.2%}), "
                    f"interp={interp} ({interp/total:.2%}), "
                    f"remaining_fail={remaining} ({remaining/total:.2%})"
                )

    if agg["files_total"] == 0:
        print("No files with usable masks were found.")
        return

    remaining = max(agg["orig_fail"] - agg["interp"], 0)
    total = agg["total_samples"]
    print(
        f"Files processed: {agg['files_total']}, missing mask files: {agg['files_missing_mask']}"
    )
    print(
        "Aggregate stats: "
        f"total_samples={total}, "
        f"orig_fail={agg['orig_fail']} ({agg['orig_fail']/total:.2%}), "
        f"interp={agg['interp']} ({agg['interp']/total:.2%}), "
        f"remaining_fail={remaining} ({remaining/total:.2%})"
    )


if __name__ == "__main__":
    main()
