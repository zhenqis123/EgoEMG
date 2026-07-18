#!/usr/bin/env python3
"""Replace one EgoEMG_incre episode with regenerated session IK labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _field_memmap(
    root: Path,
    manifest: dict,
    field_name: str,
    mode: str,
) -> np.memmap:
    field = manifest["fields"][field_name]
    return np.memmap(
        root / field["filename"],
        dtype=np.dtype(field["dtype"]),
        mode=mode,
        shape=tuple(field["shape"]),
    )


def _nearest_valid(source_ts: np.ndarray, source_valid: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_ts, target_ts, side="left")
    right = np.clip(right, 0, len(source_ts) - 1)
    left = np.clip(right - 1, 0, len(source_ts) - 1)
    use_right = np.abs(source_ts[right] - target_ts) < np.abs(target_ts - source_ts[left])
    nearest = np.where(use_right, right, left)
    return source_valid[nearest]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incre-root", type=Path, required=True)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    incre_root = args.incre_root
    session_root = args.session_root
    hand = args.hand
    hand_idx = 1 if hand == "right" else 0

    with open(incre_root / "manifest.json") as f:
        incre_manifest = json.load(f)
    incre_md = np.load(incre_root / "metadata.npz", allow_pickle=True)
    episode_ids = [
        e.decode("utf-8") if isinstance(e, bytes) else str(e)
        for e in incre_md["episode_id"]
    ]
    if args.episode_id not in episode_ids:
        raise ValueError(f"episode_id not found: {args.episode_id}")
    episode_idx = episode_ids.index(args.episode_id)
    start = int(incre_md["episode_start_idx"][episode_idx])
    length = int(incre_md["episode_length"][episode_idx])
    stop = start + length

    source_ts = np.load(session_root / "wilor_mano" / "timestamps_us.npy").astype(np.float64)
    source_valid = np.load(session_root / "memmap" / "valid.npy").astype(bool)
    with open(session_root / "memmap" / "manifest.json") as f:
        session_manifest = json.load(f)
    source_field = session_manifest["fields"][f"generated_joint_angles_{hand}"]
    source_angles = np.memmap(
        session_root / "memmap" / source_field["filename"],
        dtype=np.dtype(source_field["dtype"]),
        mode="r",
        shape=tuple(source_field["shape"]),
    )
    if len(source_ts) != source_angles.shape[0] or len(source_ts) != len(source_valid):
        raise ValueError(
            "source timestamps, angles, and valid mask must have matching lengths: "
            f"{len(source_ts)}, {source_angles.shape[0]}, {len(source_valid)}"
        )

    target_ts_mm = _field_memmap(incre_root, incre_manifest, "timestamp_us", "r")
    target_ts = np.asarray(target_ts_mm[start:stop], dtype=np.float64)
    if np.any(np.diff(source_ts) <= 0):
        order = np.argsort(source_ts, kind="stable")
        source_ts = source_ts[order]
        source_valid = source_valid[order]
        source_angles = np.asarray(source_angles)[order]

    target_angles = np.empty((length, source_angles.shape[1]), dtype=np.float32)
    source_angles_arr = np.asarray(source_angles, dtype=np.float32)
    for dim in range(source_angles_arr.shape[1]):
        target_angles[:, dim] = np.interp(
            target_ts,
            source_ts,
            source_angles_arr[:, dim],
        ).astype(np.float32)

    target_valid = _nearest_valid(source_ts, source_valid, target_ts)
    outside = (target_ts < source_ts[0]) | (target_ts > source_ts[-1])
    target_valid[outside] = False

    joint_mm = _field_memmap(incre_root, incre_manifest, f"generated_joint_angles_{hand}", "r+")
    valid_mm = _field_memmap(incre_root, incre_manifest, "generated_label_valid", "r+")
    old_angles = np.asarray(joint_mm[start:stop], dtype=np.float32)
    old_valid = np.asarray(valid_mm[start:stop, hand_idx], dtype=bool)

    diff = target_angles - old_angles
    finite = np.isfinite(target_angles).all()
    if not finite:
        raise ValueError("target_angles contains non-finite values")

    joint_mm[start:stop] = target_angles
    valid_mm[start:stop, hand_idx] = target_valid
    joint_mm.flush()
    valid_mm.flush()

    report = {
        "incre_root": str(incre_root),
        "session_root": str(session_root),
        "episode_id": args.episode_id,
        "episode_index": episode_idx,
        "start": start,
        "stop": stop,
        "length": length,
        "source_frames": int(len(source_ts)),
        "source_valid": int(source_valid.sum()),
        "source_valid_fraction": float(source_valid.mean()),
        "target_valid": int(target_valid.sum()),
        "target_valid_fraction": float(target_valid.mean()),
        "old_valid": int(old_valid.sum()),
        "old_valid_fraction": float(old_valid.mean()),
        "angle_l2_mean": float(np.linalg.norm(diff, axis=1).mean()),
        "angle_l2_p95": float(np.percentile(np.linalg.norm(diff, axis=1), 95)),
        "angle_l2_max": float(np.linalg.norm(diff, axis=1).max()),
        "target_ts_start": int(target_ts[0]),
        "target_ts_stop": int(target_ts[-1]),
        "source_ts_start": int(source_ts[0]),
        "source_ts_stop": int(source_ts[-1]),
        "finite": bool(finite),
    }
    report_path = args.report or (session_root / "memmap" / "replace_incre_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
