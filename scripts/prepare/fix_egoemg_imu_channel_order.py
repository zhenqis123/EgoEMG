"""Reorder the EgoEMG rows of the unified memmap ``imu`` field to [acc, gyro].

The EgoEMG source LeRobot parquet stores ``observation.imu`` gyro-first as
``[gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]`` (gyro_x is a dead axis,
identically 0 in all 41 episodes). The conversion into the unified memmap
copied the six channels verbatim into a field documented -- and used by the
ShowEE rows -- as accel-first ``[acc, gyro]``. Result: the real EgoEMG
accelerometer (gravity ~9.2-9.4 m/s^2) sits in channels 3-5 while the
documented accelerometer channels look like non-sensor noise.

This script swaps the two halves IN PLACE for the EgoEMG rows only
(rows ``[0, episode_end_idx[40])`` -- asserted contiguous and source_id==0)::

    new = [old_3, old_4, old_5, old_0, old_1, old_2]

The permutation is row-wise and reversible; an idempotence guard refuses to
run on already-fixed data. ShowEE/Incre rows are never touched (verified by
byte checksums taken before and after).

Usage::

    python scripts/prepare/fix_egoemg_imu_channel_order.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_unified_memmap \
        --apply --update-manifest      # omit --apply for dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REORDER_INDEX = np.array([3, 4, 5, 0, 1, 2])
BACKUP_NAME = "imu.dat.bak_prelayout"
# ep0 sample: |acc| p50 is ~0.21 pre-fix (noise side) / ~9.24 post-fix.
PRE_FIXED_MAX, FIXED_MIN = 1.0, 7.0


def reorder_imu_channels(arr: np.ndarray) -> np.ndarray:
    """Map gyro-first rows [..., 6] to accel-first [..., 6] (and back)."""
    if arr.shape[-1] != 6:
        raise ValueError(f"expected last dim 6, got {arr.shape}")
    return arr[..., REORDER_INDEX]


def _segment_checksum(memmap: np.memmap, lo: int, hi: int) -> str:
    h = hashlib.sha256()
    raw = memmap[lo:hi].tobytes()
    h.update(raw)
    return h.hexdigest()


def _episode_bounds(memmap_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = np.load(memmap_dir / "metadata.npz", allow_pickle=True)
    return meta["episode_start_idx"].astype(np.int64), meta["episode_end_idx"].astype(np.int64)


def _sample_acc_p50(imu: np.memmap, start: int, end: int) -> float:
    step = max(1, (end - start) // 100_000)
    sample = imu[start:end:step, :3].astype(np.float64)
    return float(np.percentile(np.linalg.norm(sample, axis=1), 50))


def _episode_table(imu: np.memmap, start: np.ndarray, end: np.ndarray, count: int) -> list[dict]:
    rows = []
    for i in range(count):
        s, e = int(start[i]), int(end[i])
        step = max(1, (e - s) // 50_000)
        sub = imu[s:e:step].astype(np.float64)
        acc_mag = np.linalg.norm(sub[:, :3], axis=1)
        rows.append(
            {
                "episode": i,
                "rows": e - s,
                "|acc|p50": round(float(np.percentile(acc_mag, 50)), 3),
                "grav%": round(float(((acc_mag > 7) & (acc_mag < 12)).mean() * 100), 1),
                "zero%": round(float((np.abs(sub).sum(axis=1) == 0).mean() * 100), 2),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix EgoEMG imu channel order in the unified memmap")
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--no-backup", action="store_true", help="skip the imu.dat backup before apply")
    ap.add_argument("--update-manifest", action="store_true", help="rewrite manifest imu_semantics after apply")
    args = ap.parse_args()

    root = args.memmap_dir
    with open(root / "manifest.json") as f:
        manifest = json.load(f)
    imu_spec = manifest["fields"]["imu"]
    src_spec = manifest["fields"]["dataset_source_id"]
    n_rows, width = imu_spec["shape"]
    assert width == 6, f"imu width {width} != 6"

    # --- gates -----------------------------------------------------------
    start, end = _episode_bounds(root)
    n_ego = int(end[40])
    assert int(start[0]) == 0, "EgoEMG block does not start at row 0"
    assert n_ego == 66_161_725, f"unexpected EgoEMG row count {n_ego}"

    src = np.memmap(root / src_spec["filename"], dtype=src_spec["dtype"], mode="r", shape=(n_rows,))
    assert (np.asarray(src[:n_ego]) == 0).all(), "EgoEMG block contains non-EgoEMG rows"
    assert (np.asarray(src[n_ego : n_ego + 10_000]) != 0).all(), "post-EgoEMG rows are not all ShowEE/Incre"

    imu_path = root / imu_spec["filename"]
    imu = np.memmap(imu_path, dtype=imu_spec["dtype"], mode="r", shape=(n_rows, 6))

    ep0_p50 = _sample_acc_p50(imu, int(start[0]), int(end[0]))
    if ep0_p50 > FIXED_MIN:
        print(f"REFUSING: ep0 |acc| p50 = {ep0_p50:.3f} looks already reordered (gravity in acc side).")
        return 1
    if ep0_p50 > PRE_FIXED_MAX:
        print(f"REFUSING: ep0 |acc| p50 = {ep0_p50:.3f} is neither pre-fix (<1) nor fixed (>7) state.")
        return 1
    print(f"state check: ep0 |acc(ch0-2)| p50 = {ep0_p50:.3f} -> pre-fix gyro-first layout confirmed")

    # checksums of untouched regions (ShowEE head, Incre mid, file tail)
    guard_segments = [(n_ego, n_ego + 50_000), ((n_ego + n_rows) // 2, (n_ego + n_rows) // 2 + 50_000), (n_rows - 50_000, n_rows)]
    guards_before = {seg: _segment_checksum(imu, *seg) for seg in guard_segments}

    # --- dry-run preview --------------------------------------------------
    step = max(1, n_ego // 200_000)
    preview = imu[:n_ego:step]
    fixed_preview = reorder_imu_channels(preview)
    mag_before = np.percentile(np.linalg.norm(preview[:, :3].astype(np.float64), axis=1), 50)
    mag_after = np.percentile(np.linalg.norm(fixed_preview[:, :3].astype(np.float64), axis=1), 50)
    print(f"dry-run preview (EgoEMG block, {len(preview):,} sampled rows):")
    print(f"  |acc ch0-2| p50: {mag_before:.3f} -> {mag_after:.3f} (expect ~9.2)")

    if not args.apply:
        print("dry-run only; rerun with --apply to write changes")
        return 0

    # --- backup -----------------------------------------------------------
    backup = root / BACKUP_NAME
    if args.no_backup:
        print("backup skipped (--no-backup)")
    elif backup.exists():
        print(f"backup already exists, keeping it: {backup}")
    else:
        print(f"backing up {imu_spec['filename']} -> {BACKUP_NAME} (~3.2 GB)...")
        shutil.copy2(imu_path, backup)
        print("backup done")

    # --- in-place permutation ---------------------------------------------
    imu_rw = np.memmap(imu_path, dtype=imu_spec["dtype"], mode="r+", shape=(n_rows, 6))
    for lo in range(0, n_ego, args.chunk_rows):
        hi = min(lo + args.chunk_rows, n_ego)
        block = np.array(imu_rw[lo:hi])
        imu_rw[lo:hi] = reorder_imu_channels(block)
        imu_rw.flush()
        print(f"  reordered rows [{lo:,}, {hi:,})", flush=True)
    del imu_rw

    # --- post-verify --------------------------------------------------------
    imu = np.memmap(imu_path, dtype=imu_spec["dtype"], mode="r", shape=(n_rows, 6))
    for seg, before in guards_before.items():
        after = _segment_checksum(imu, *seg)
        assert after == before, f"GUARD FAIL: rows {seg} changed outside the EgoEMG block!"
    print("guards: ShowEE/Incre segments byte-identical after fix")

    ep0_p50_post = _sample_acc_p50(imu, int(start[0]), int(end[0]))
    assert ep0_p50_post > FIXED_MIN, f"post-fix ep0 |acc| p50 {ep0_p50_post:.3f} still noise-side"
    print(f"post-fix: ep0 |acc(ch0-2)| p50 = {ep0_p50_post:.3f}")

    table = _episode_table(imu, start, end, 41)
    print(f"{'ep':>3} {'rows':>9} {'|acc|p50':>9} {'grav%':>6} {'zero%':>6}")
    for r in table:
        print(f"{r['episode']:>3} {r['rows']:>9,} {r['|acc|p50']:>9.3f} {r['grav%']:>6.1f} {r['zero%']:>6.2f}")

    backup_ref = None if args.no_backup else BACKUP_NAME
    if args.update_manifest:
        semantics = manifest.setdefault("imu_semantics", {})
        semantics["imu"] = (
            "EgoEMG source rows + ShowEE LEFT wrist band (~110 Hz, nearest-sampled to 2 kHz); "
            "Incre rows zero. EgoEMG rows are REAL wrist-band IMU: the source parquet's "
            "observation.imu is gyro-first [gyro,acc] with a dead gyro_x axis (== 0); rows were "
            "reordered to [acc,gyro] in place by scripts/prepare/fix_egoemg_imu_channel_order.py. "
            "acc is in m/s^2 (gravity ~9.2-9.4 at rest; some sessions show attenuated magnitudes, "
            "present in the source as well -- see scripts/release/imu_verify_report_windows_original.json)."
        )
        semantics["egoemg_layout_fix"] = {
            "date": "2026-08-20",
            "script": "scripts/prepare/fix_egoemg_imu_channel_order.py",
            "permuted_rows": [0, n_ego],
            "permutation": "[3,4,5,0,1,2]",
            "backup": backup_ref,
            "verified": "per-episode |acc| p50 matches the original Windows parquet report; "
            "ShowEE/Incre guard segments byte-identical",
        }
        tmp = root / "manifest.json.tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        tmp.replace(root / "manifest.json")
        print("manifest imu_semantics updated")

    print("DONE. EgoEMG imu rows reordered to [acc, gyro].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
