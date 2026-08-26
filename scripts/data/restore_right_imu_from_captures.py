#!/usr/bin/env python3
"""Restore the right-wrist IMU (`imu_band_right`) for EgoEMG episodes.

The capture pipeline (step1_raw_to_intermediate.py) loaded the wristband IMU
only from the PRIMARY (left) device, so the right band's weili_imu.csv —
recorded on disk for 39/41 EgoEMG sessions — never reached the lerobot
parquets or the memmap, leaving `imu_band_right` zero-filled for episodes
0..40. This script re-ingests those CSVs and linearly interpolates them onto
each episode's `timestamp_us` grid, mirroring the left-band semantics
(`legacy.linear_interp_matrix`). ShowEE/Incre rows are untouched.

Inputs: a mapping of episode -> session CSV (see --map-json), produced by
matching each episode's EMG timestamp range against the capture sessions.

Usage:
  python scripts/data/restore_right_imu_from_captures.py \
      --memmap-dir data/EgoEMG_full_memmap \
      --map-json /tmp/ep_session_map.json \
      --csv-root /home/xiziheng/develop/right_imu_recovery
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t, v = [], []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[0].strip().lower().startswith("timestamp"):
            pass  # normal header
        for row in reader:
            try:
                ts = int(row[0])
                vals = [float(x) for x in row[1:7]]
            except (ValueError, IndexError):
                continue  # truncated/partial line
            if len(vals) == 6:
                t.append(ts)
                v.append(vals)
    return np.asarray(t, dtype=np.int64), np.asarray(v, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--map-json", type=Path, required=True)
    ap.add_argument("--csv-root", type=Path, required=True)
    args = ap.parse_args()

    mapping = json.loads(args.map_json.read_text())
    manifest = json.loads((args.memmap_dir / "manifest.json").read_text())
    F = manifest["fields"]

    ts = np.memmap(args.memmap_dir / F["timestamp_us"]["filename"],
                   dtype=F["timestamp_us"]["dtype"], mode="r")
    ep = np.memmap(args.memmap_dir / F["episode_index"]["filename"],
                   dtype=F["episode_index"]["dtype"], mode="r")
    spec = F["imu_band_right"]
    field = np.memmap(args.memmap_dir / spec["filename"],
                      dtype=spec["dtype"], mode="r+", shape=tuple(spec["shape"]))

    report = []
    for ep_str, info in sorted(mapping.items(), key=lambda kv: int(kv[0])):
        e = int(ep_str)
        sess = info["session"].split("/")[-1]
        csv_path = args.csv_root / sess / "right_weili_imu.csv"
        rows = np.where(ep == e)[0]
        if not csv_path.is_file():
            report.append((e, sess, len(rows), 0.0, "no csv (stays zero)"))
            continue
        imu_t, imu_v = load_imu_csv(csv_path)
        if len(imu_t) < 2:
            report.append((e, sess, len(rows), 0.0, f"degenerate csv ({len(imu_t)} rows)"))
            continue
        t_rows = np.asarray(ts[rows], dtype=np.int64)
        # np.interp clamps outside the IMU range, same edge behaviour as the
        # left-band pipeline (linear hold at the ends).
        out = np.stack([
            np.interp(t_rows, imu_t, imu_v[:, c]) for c in range(6)
        ], axis=1).astype(spec["dtype"])
        # Raw CSVs are gyro-first [gyro xyz, acc xyz]; the memmap field is
        # canonical accel-first [acc, gyro] (see
        # scripts/prepare/fix_egoemg_imu_channel_order.py).
        out = out[:, [3, 4, 5, 0, 1, 2]].copy()
        covered = ((t_rows >= imu_t[0]) & (t_rows <= imu_t[-1])).mean()
        field[rows] = out
        report.append((e, sess, len(rows), float(out.std()), f"covered {covered*100:.1f}%"))

    field.flush()
    print(f"{'ep':>3} {'session':>26} {'rows':>9} {'std':>7}  note")
    for e, sess, n, std, note in report:
        print(f"{e:>3} {sess:>26} {n:>9} {std:>7.3f}  {note}")
    fixed = sum(1 for r in report if r[3] > 0)
    print(f"\nrestored right IMU for {fixed}/{len(report)} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
