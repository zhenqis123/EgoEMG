"""Group the unified memmap's flat .dat files into modality subdirectories.

Pure filesystem-level reorganization: every file keeps its byte content and
its field name; only its path (and the manifest's ``filename`` pointer)
changes, so manifest-driven consumers are unaffected. Hardcoded-path
consumers are resolved through the manifest (see center_frame.py).

Layout (root keeps manifest.json, metadata.npz, gesture_classes.json,
checksums.json only):

  core/        episode_index frame_index timestamp_us subject_id
               dataset_source_id frame_split_id is_first is_last
  emg/         emg_{left,right}_{raw,filtered_paper}
  imu/         imu_band_left imu_band_right imu_cam_head imu_cam_wrist_{l,r}
  labels/      label_gesture_class label_gesture_active generated_label_valid
               generated_joint_angles_{left,right}
  mano/        generated_mano_{left,right}_{pose,beta}
               mocap_mano_{left,right}_world_transform
  mocap_hands/ mocap_{left,right}_{keypoints,valid}
  mocap_wrist/ mocap_{left,right}_wrist_* families
  mocap_head/  mocap_head_{position,orientation,valid,rigid_markers,transform}
  vision/      image_{zed,head,wrist_left,wrist_right}_{frame_index,stale,delta_ms}

Usage:
  python scripts/data/group_memmap_layout.py --memmap-dir <dir> [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def group_for(field: str) -> str:
    if field.startswith(("episode_index", "frame_index", "timestamp_us", "subject_id",
                         "dataset_source_id", "frame_split_id", "is_first", "is_last")):
        return "core"
    if field.startswith("emg_"):
        return "emg"
    if field.startswith("imu_"):
        return "imu"
    if field.startswith("label_") or field == "generated_label_valid" \
            or field.startswith("generated_joint_angles_"):
        return "labels"
    if field.startswith("generated_mano_") or field.startswith("mocap_mano_"):
        return "mano"
    if field.startswith(("mocap_left_keypoints", "mocap_right_keypoints",
                         "mocap_left_valid", "mocap_right_valid")):
        return "mocap_hands"
    if field.startswith(("mocap_left_wrist", "mocap_right_wrist")):
        return "mocap_wrist"
    if field.startswith("mocap_head"):
        return "mocap_head"
    if field.startswith("image_"):
        return "vision"
    raise ValueError(f"no group rule for field {field!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Group flat .dat files into modality subdirectories")
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    root = args.memmap_dir
    manifest_path = root / "manifest.json"
    m = json.loads(manifest_path.read_text())

    if m.get("layout") == "grouped":
        print("already grouped; nothing to do")
        return 0
    if any("/" in spec["filename"] for spec in m["fields"].values()):
        print("REFUSING: manifest already contains subdirectory filenames")
        return 1

    moves: list[tuple[str, str]] = []
    for section in ("fields", "episode_fields"):
        for name, spec in m[section].items():
            group = group_for(name)
            new_rel = f"{group}/{spec['filename']}"
            moves.append((spec["filename"], new_rel))
            print(f"  planned: {spec['filename']} -> {new_rel}")
    if not args.apply:
        print("dry-run only; rerun with --apply")
        return 0

    for group in {rel.split("/")[0] for _, rel in moves}:
        (root / group).mkdir(exist_ok=True)
    for old_rel, new_rel in moves:
        (root / old_rel).rename(root / new_rel)

    for section in ("fields", "episode_fields"):
        for name, spec in m[section].items():
            spec["filename"] = f"{group_for(name)}/{name}.dat"
    m["layout"] = "grouped"
    m.setdefault("schema_migration", {})["ops"].append(
        "grouped flat .dat files into modality subdirectories (content untouched)"
    )
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(manifest_path)

    ck_path = root / "checksums.json"
    if ck_path.exists():
        move_map = dict(moves)
        ck = {move_map.get(k, k): v for k, v in json.loads(ck_path.read_text()).items()}
        ck_path.write_text(json.dumps(ck, indent=2))
        print(f"checksums.json keys remapped ({len(ck)} entries)")

    print(f"grouped {len(moves)} files into {len({r.split('/')[0] for _, r in moves})} directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
