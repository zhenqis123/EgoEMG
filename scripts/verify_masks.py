#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from emg2pose.datasets.emg2pose_dataset import Emg2PoseSessionData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify IK masks: check lengths, complementarity, and NaN coverage."
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
        "--dry-run",
        action="store_true",
        help="Only report issues; do not stop on first error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    session_paths = sorted(data_root.rglob(args.pattern))
    if not session_paths:
        print(f"No session files found under {data_root} with pattern {args.pattern}")
        return

    issues = []
    for session_path in tqdm(session_paths, desc="Verifying masks"):
        try:
            with h5py.File(session_path, "r") as f:
                if Emg2PoseSessionData.HDF5_GROUP not in f:
                    issues.append((session_path, "missing_group"))
                    continue
                g = f[Emg2PoseSessionData.HDF5_GROUP]
                ts = g[Emg2PoseSessionData.TIMESERIES]
                T = len(ts)

                # Check presence and length
                if "no_ik_failure" not in g or "ik_failure_mask" not in g:
                    issues.append((session_path, "mask_missing"))
                    continue
                no_mask = g["no_ik_failure"][...].astype(bool)
                ik_mask = g["ik_failure_mask"][...].astype(bool)
                if len(no_mask) != T or len(ik_mask) != T:
                    issues.append((session_path, "mask_length_mismatch"))
                    continue

                # Check complementarity
                if not np.array_equal(no_mask, ~ik_mask):
                    issues.append((session_path, "mask_not_complement"))

                # Check interpolated_mask length if present
                interp = None
                if "interpolated_mask" in g:
                    interp = g["interpolated_mask"][...].astype(bool)
                    if len(interp) != T:
                        issues.append((session_path, "interp_length_mismatch"))

                # Check NaNs in joint_angles outside mask
                joint_angles = ts[Emg2PoseSessionData.JOINT_ANGLES]
                nan_steps = np.any(~np.isfinite(joint_angles), axis=1)
                # Effective invalid (should be masked) = IK failure OR NaN
                invalid = ik_mask | nan_steps
                if interp is not None:
                    invalid = invalid & ~interp  # interpolated positions are allowed
                uncovered_nan = nan_steps & ~invalid
                if uncovered_nan.any():
                    issues.append((session_path, f"nan_unmasked_count={uncovered_nan.sum()}"))

        except Exception as exc:  # noqa: BLE001
            issues.append((session_path, f"exception: {exc}"))
            if not args.dry_run:
                break

    if not issues:
        print("All files passed mask checks.")
        return

    print(f"Found {len(issues)} issues:")
    for path, msg in issues:
        print(f"- {path}: {msg}")


if __name__ == "__main__":
    main()
