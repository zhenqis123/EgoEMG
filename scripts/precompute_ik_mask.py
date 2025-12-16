#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from emg2pose.datasets.emg2pose_dataset import Emg2PoseSessionData
from emg2pose.utils import get_contiguous_ones, get_ik_failures_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute IK failure mask and store it inside session HDF5 files "
            "to avoid recomputing during training."
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
        "--overwrite",
        action="store_true",
        help="Recompute and replace existing no_ik_failure dataset if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be updated without writing changes.",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=0.5,
        help=(
            "Do not interpolate gaps longer than this duration; leave them as NaN. "
            "Default: 0.5s."
        ),
    )
    parser.add_argument(
        "--recompute-mask",
        action="store_true",
        help=(
            "Force recomputing IK failure mask from current joint_angles. "
            "默认复用已有 ik_failure_mask/no_ik_failure 数据集以避免插值后被覆盖。"
        ),
    )
    return parser.parse_args()


def _mask_dataset_ok(group: h5py.Group, name: str, expected_len: int) -> bool:
    return name in group and group[name].shape[0] == expected_len


def needs_update(
    group: h5py.Group, expected_len: int, overwrite: bool, mask_was_computed: bool
) -> bool:
    if overwrite:
        return True
    if mask_was_computed:
        return True
    has_no = _mask_dataset_ok(group, "no_ik_failure", expected_len)
    has_ik = _mask_dataset_ok(group, "ik_failure_mask", expected_len)
    return not (has_no and has_ik)


def interpolate_joint_angles(
    joint_angles: np.ndarray,
    valid_mask: np.ndarray,
    timestamps: np.ndarray,
    max_gap_seconds: float | None,
) -> tuple[np.ndarray, bool, np.ndarray, dict[str, int]]:
    """Interpolate joint angles where valid_mask is False, with guards:

    1. If a gap has no valid point on either side (all zeros before/after), keep NaN.
    2. If the time span across the gap exceeds max_gap_seconds, keep NaN.

    Returns the filled array, a flag indicating whether any values changed,
    a boolean mask of interpolated positions, and per-call statistics.
    """
    if valid_mask.all():
        empty_stats = {
            "gaps_total": 0,
            "gaps_filled": 0,
            "gaps_kept_no_anchor": 0,
            "gaps_kept_long": 0,
            "samples_filled": 0,
            "samples_kept": 0,
        }
        interp_mask = np.zeros_like(valid_mask, dtype=bool)
        return joint_angles, False, interp_mask, empty_stats

    filled = joint_angles.astype(np.float64, copy=True)
    updated = False
    interp_mask = np.zeros_like(valid_mask, dtype=bool)

    valid_idx = np.where(valid_mask)[0]
    failure_blocks = get_contiguous_ones(~valid_mask)

    stats = {
        "gaps_total": len(failure_blocks),
        "gaps_filled": 0,
        "gaps_kept_no_anchor": 0,
        "gaps_kept_long": 0,
        "samples_filled": 0,
        "samples_kept": 0,
    }

    # Early exit: no valid anchor anywhere -> mark failures as NaN and skip.
    if len(failure_blocks) and not valid_idx.size:
        for start, end in failure_blocks:
            filled[start : end + 1] = np.nan
            stats["samples_kept"] += end - start + 1
        stats["gaps_kept_no_anchor"] = len(failure_blocks)
        return filled, True, interp_mask, stats

    for start, end in failure_blocks:
        left_candidates = valid_idx[valid_idx < start]
        right_candidates = valid_idx[valid_idx > end]

        # Rule 1: missing anchor on either side -> keep NaN.
        if not left_candidates.size or not right_candidates.size:
            filled[start : end + 1] = np.nan
            updated = True
            stats["gaps_kept_no_anchor"] += 1
            stats["samples_kept"] += end - start + 1
            continue

        left_idx = left_candidates[-1]
        right_idx = right_candidates[0]

        # Rule 2: gap duration too long -> keep NaN.
        if max_gap_seconds is not None:
            gap_duration = timestamps[right_idx] - timestamps[left_idx]
            if gap_duration > max_gap_seconds:
                filled[start : end + 1] = np.nan
                updated = True
                stats["gaps_kept_long"] += 1
                stats["samples_kept"] += end - start + 1
                continue

        t_block = np.arange(start, end + 1)
        anchors = np.array([left_idx, right_idx])
        for j in range(filled.shape[1]):
            filled[start : end + 1, j] = np.interp(
                t_block, anchors, filled[anchors, j]
            )
        interp_mask[start : end + 1] = True
        updated = True
        stats["gaps_filled"] += 1
        stats["samples_filled"] += end - start + 1

    return filled, updated, interp_mask, stats


def load_masks(
    group: h5py.Group,
    joint_angles: np.ndarray,
    expected_len: int,
    recompute_mask: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load masks from file when available, otherwise compute from joint_angles."""
    if not recompute_mask:
        if _mask_dataset_ok(group, "ik_failure_mask", expected_len):
            ik_failure_mask = group["ik_failure_mask"][...].astype(bool)
            no_ik_mask = ~ik_failure_mask
            return no_ik_mask, ik_failure_mask, False
        if _mask_dataset_ok(group, "no_ik_failure", expected_len):
            no_ik_mask = group["no_ik_failure"][...].astype(bool)
            ik_failure_mask = ~no_ik_mask
            return no_ik_mask, ik_failure_mask, False

    no_ik_mask = get_ik_failures_mask(joint_angles).astype(np.bool_)
    ik_failure_mask = ~no_ik_mask
    return no_ik_mask, ik_failure_mask, True


def write_mask(
    session_path: Path,
    overwrite: bool,
    dry_run: bool,
    max_gap_seconds: float | None,
    recompute_mask: bool,
) -> tuple[str, dict[str, int]]:
    with h5py.File(session_path, "r+") as f:
        if Emg2PoseSessionData.HDF5_GROUP not in f:
            return "missing_group", {}
        group = f[Emg2PoseSessionData.HDF5_GROUP]

        timeseries = group[Emg2PoseSessionData.TIMESERIES]
        expected_len = len(timeseries)

        joint_angles = timeseries[Emg2PoseSessionData.JOINT_ANGLES]
        no_ik_mask, ik_failure_mask, mask_was_computed = load_masks(
            group, joint_angles, expected_len, recompute_mask
        )
        should_update_masks = needs_update(group, expected_len, overwrite, mask_was_computed)
        timestamps = timeseries[Emg2PoseSessionData.TIMESTAMPS]

        filled_joint_angles, angles_updated, interp_mask, interp_stats = interpolate_joint_angles(
            joint_angles, no_ik_mask, timestamps, max_gap_seconds
        )

        if dry_run:
            status = "pending" if (angles_updated or should_update_masks) else "skipped"
            return status, interp_stats

        updated = False
        if angles_updated:
            timeseries[Emg2PoseSessionData.JOINT_ANGLES] = filled_joint_angles
            updated = True

        if should_update_masks:
            for name in ("no_ik_failure", "ik_failure_mask", "interpolated_mask"):
                if name in group:
                    del group[name]

            ds_no = group.create_dataset("no_ik_failure", data=no_ik_mask)
            ds_no.attrs["description"] = (
                "Boolean mask where True indicates no IK failure; "
                "computed from joint_angles before interpolation."
            )
            ds_ik = group.create_dataset("ik_failure_mask", data=ik_failure_mask)
            ds_ik.attrs["description"] = (
                "Boolean mask where True indicates original IK failure (zeros) before interpolation."
            )
            ds_interp = group.create_dataset("interpolated_mask", data=interp_mask.astype(np.bool_))
            ds_interp.attrs["description"] = (
                "Boolean mask where True indicates samples filled by interpolation due to IK failure gaps."
            )
            updated = True

        return ("written" if updated else "skipped"), interp_stats


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    session_paths = sorted(data_root.rglob(args.pattern))
    if not session_paths:
        print(f"No session files found under {data_root} with pattern {args.pattern}")
        return

    counts: dict[str, int] = {"written": 0, "skipped": 0, "pending": 0, "missing_group": 0}
    agg_stats: dict[str, int] = {
        "gaps_total": 0,
        "gaps_filled": 0,
        "gaps_kept_no_anchor": 0,
        "gaps_kept_long": 0,
        "samples_filled": 0,
        "samples_kept": 0,
    }
    for session_path in tqdm(session_paths, desc="Precomputing IK masks"):
        status, interp_stats = write_mask(
            session_path,
            args.overwrite,
            args.dry_run,
            args.max_gap_seconds,
            args.recompute_mask,
        )
        counts[status] = counts.get(status, 0) + 1
        for k, v in interp_stats.items():
            agg_stats[k] = agg_stats.get(k, 0) + int(v)

    print(
        f"Done. written={counts['written']}, skipped={counts['skipped']}, "
        f"pending={counts['pending']}, missing_group={counts['missing_group']}"
    )
    print(
        "Interpolation stats: "
        f"gaps_total={agg_stats['gaps_total']}, gaps_filled={agg_stats['gaps_filled']}, "
        f"kept_no_anchor={agg_stats['gaps_kept_no_anchor']}, kept_long={agg_stats['gaps_kept_long']}, "
        f"samples_filled={agg_stats['samples_filled']}, samples_kept={agg_stats['samples_kept']}"
    )


if __name__ == "__main__":
    main()
