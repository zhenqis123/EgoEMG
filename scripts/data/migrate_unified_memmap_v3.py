"""Migrate a unified EgoEMG memmap from schema v2 to v3.

v3 redesign (maintainer-approved breaking changes, 2026-08-20):

Renames (self-describing IMU taxonomy; head tracked -> valid):
  imu               -> imu_band_left     (EgoEMG band + ShowEE LEFT wrist band)
  imu_right         -> imu_band_right    (ShowEE RIGHT wrist band + Incre band)
  imu_head          -> imu_cam_head      (ShowEE head camera IMU, ~420 Hz)
  imu_wrist_left    -> imu_cam_wrist_left  (ShowEE left wrist camera IMU)
  imu_wrist_right   -> imu_cam_wrist_right (ShowEE right wrist camera IMU)
  mocap_head_tracked -> mocap_head_valid

Drops (dead or misleading):
  task_index        (constant 0 across all rows)
  source_index      (row index duplicate of frame_index)
  is_terminal       (robotic-episode vestige, meaningless for gesture data)
  emg_right_filtered(orphan column: Incre-only, no left counterpart, pipeline
                      unreproducible -- see docs/data_known_issues.md #2/#9)
  timestamp         (float64 seconds copy; timestamp_us is authoritative)

Validity (design revision after full-dataset probing):
  mocap_{left,right}_valid KEEP their (N, 21) per-keypoint shape — on ShowEE
  rows 74.4% of frames are partially valid (occlusion), so the per-keypoint
  axis is genuine information. A scalar collapse was considered and rejected.

Data fix (idempotent safety, docs/data_known_issues.md #19):
  Incre rows (dataset_source_id == 2): mocap_{left,right}_valid forced to
  False. Current data already measures all-False there; the step stays as a
  guard so the policy in the manifest can never drift from the bits.

Manifest:
  format_version -> egoemg_v3_memmap; fields table rebuilt; new comprehensive
  `field_semantics` key; imu_semantics updated to the new names; a
  `schema_migration` record is appended.

Usage:
  python scripts/data/migrate_unified_memmap_v3.py \
    --memmap-dir /path/to/EgoEMG_unified_memmap [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RENAMES = {
    "imu": "imu_band_left",
    "imu_right": "imu_band_right",
    "imu_head": "imu_cam_head",
    "imu_wrist_left": "imu_cam_wrist_left",
    "imu_wrist_right": "imu_cam_wrist_right",
    "mocap_head_tracked": "mocap_head_valid",
}
DROPS = ["task_index", "source_index", "is_terminal", "emg_right_filtered", "timestamp"]
# mocap_{left,right}_valid intentionally keep (N, 21): per-keypoint validity
# is genuine information on ShowEE rows (74% partially valid).

FIELD_SEMANTICS = {
    "emg": (
        "All EMG fields are (N, 8) float32 in millivolts (mV), converted from "
        "source uV/ADC counts by /1000 in the v2 converter. *_raw is "
        "unfiltered; *_filtered_paper is the per-episode FFT filter "
        "(20-850 Hz soft band, 50/100 Hz notch) documented in "
        "scripts/data/filter_emg_into_new_columns.py."
    ),
    "imu": (
        "All imu_* fields are (N, 6) float32 [acc_x, acc_y, acc_z, gyro_x, "
        "gyro_y, gyro_z]; acc in m/s^2 (gravity ~9.2-9.7 at rest), gyro as "
        "stored by the band. imu_band_left additionally covers the EgoEMG "
        "source rows whose gyro_x axis is dead (stored 0). Incre's band acc "
        "was scaled by 9.8/5.4 to the m/s^2 convention. Camera IMUs "
        "(imu_cam_*) are ~420 Hz; wrist bands are ~110 Hz, nearest-sampled "
        "onto the 2 kHz row grid."
    ),
    "mocap_positions": (
        "mocap_*_position are (N, 3) float32 in meters (ShowEE sources "
        "divided by 1000 from mm); mocap_*_keypoints are (N, 21, 3) MANO "
        "topology keypoints; mocap_*_rigid_markers are (N, K, 3) tracker "
        "marker positions with K = 5 for the left wrist rig and K = 4 for "
        "head/right-wrist rigs (hardware configuration difference)."
    ),
    "mocap_orientation": (
        "mocap_*_orientation are (N, 4) float32 quaternions, stored as "
        "exported by the mocap system's h5 'quaternion' dataset."
    ),
    "validity": (
        "mocap_{left,right}_valid are (N, 21) per-keypoint bools — partially "
        "valid frames are common on ShowEE (occlusion; 74% of rows) and "
        "essentially absent on EgoEMG. mocap_{left,right}_wrist_angles_valid "
        "gate the pitch/yaw scalars; mocap_head_valid gates the head rig; "
        "generated_label_valid (N, 2) gates joint-angle labels per hand. "
        "Incre rows carry all-False validity with zero mocap data by policy."
    ),
    "timestamps": (
        "timestamp_us (int64, microseconds since epoch) is authoritative; "
        "v3 dropped the redundant float64 seconds copy."
    ),
    "gesture": (
        "label_gesture_class uses -1 for 'no active gesture'; "
        "label_gesture_active is the boolean companion."
    ),
    "video_sync": (
        "Each image_{stream} carries frame_index (-1 = unavailable), "
        "delta_ms (frame age relative to the row timestamp), and stale "
        "(True = no usable frame)."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate unified memmap schema v2 -> v3")
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    root = args.memmap_dir
    manifest_path = root / "manifest.json"
    m = json.loads(manifest_path.read_text())

    if m.get("format_version") == "egoemg_v3_memmap":
        print("already at v3; nothing to do")
        return 0
    if m.get("format_version") != "egoemg_v2_memmap":
        print(f"REFUSING: unexpected format_version {m.get('format_version')!r}")
        return 1

    fields = m["fields"]
    for name in RENAMES:
        assert name in fields, f"missing expected field {name}"
    for name in DROPS:
        assert name in fields, f"missing expected field {name}"

    ops: list[str] = []
    plan = {**{f"rename {k} -> {v}": None for k, v in RENAMES.items()},
            **{f"drop {k}": None for k in DROPS},
            "fix Incre mocap valid bits": None,
            "manifest: v3 fields + field_semantics": None}
    for step in plan:
        print(f"  planned: {step}")
    if not args.apply:
        print("dry-run only; rerun with --apply")
        return 0


    # 1) renames (cheap os-level)
    for old, new in RENAMES.items():
        spec = fields.pop(old)
        (root / spec["filename"]).rename(root / f"{new}.dat")
        fields[new] = {**spec, "filename": new + ".dat"}
        ops.append(f"renamed {old} -> {new}")

    # 2) drops
    for name in DROPS:
        (root / fields[name]["filename"]).unlink()
        del fields[name]
        ops.append(f"dropped {name}")

    # 3) Incre valid-bit fix (per-keypoint arrays stay (N, 21))
    src_spec = fields["dataset_source_id"]
    src = np.asarray(np.memmap(root / src_spec["filename"], dtype=src_spec["dtype"],
                               mode="r", shape=tuple(src_spec["shape"])))
    incre = src == 2
    fixed = 0
    for name in ("mocap_left_valid", "mocap_right_valid"):
        spec = fields[name]
        arr = np.memmap(root / spec["filename"], dtype=spec["dtype"], mode="r+",
                        shape=tuple(spec["shape"]))
        hit = incre[:, None] & arr
        fixed += int(hit.sum())
        arr[incre] = False
        arr.flush()
        del arr
    ops.append(f"cleared {fixed} Incre per-keypoint valid entries")

    # 5) manifest rebuild
    m["format_version"] = "egoemg_v3_memmap"
    m["field_semantics"] = FIELD_SEMANTICS
    if "imu_semantics" in m:
        for old, new in RENAMES.items():
            if old in m["imu_semantics"]:
                m["imu_semantics"][new] = m["imu_semantics"].pop(old)
    m["source_policies"] = {
        k: (v + " [v3: mocap valid bits corrected to False on zero-data rows]"
            if k == "egoemg_incre" else v)
        for k, v in m["source_policies"].items()
    }
    m["schema_migration"] = {
        "from": "egoemg_v2_memmap",
        "to": "egoemg_v3_memmap",
        "date": "2026-08-20",
        "script": "scripts/data/migrate_unified_memmap_v3.py",
        "ops": ops,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(manifest_path)
    print(f"manifest rewritten (v3), {len(ops)} ops applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
