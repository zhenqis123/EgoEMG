"""Rename the "webcam" stream to "head view" in the release memmap.

The webcam camera is a head-mounted camera, so all of its schema names
are renamed to the ``head`` view:

  * frame fields ``image_webcam_frame_index/stale/delta_ms``
  * mocap rigid-body fields ``mocap_webcam_*`` (the head-mounted
    camera's pose)
  * metadata key ``episode_webcam_video_path`` -> ``episode_head_video_path``
  * the 63 all-intra videos ``episode_XXXXXX_allintra.mp4`` ->
    ``episode_XXXXXX_head_allintra.mp4`` (and the metadata path values
    ``episode_XXXXXX.mp4`` -> ``episode_XXXXXX_head.mp4``)

``.dat`` contents are unchanged (only file names and manifest entries
are rewritten).  All changes are backed up.

Usage::

    python scripts/prepare/rename_webcam_to_head.py \
        --memmap-dir /mnt/nvme/xiziheng/EgoEMG_full_memmap \
        --allintra-root /mnt/nvme/xiziheng/EgoEMG_videos
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

FIELD_RENAMES = {
    "image_webcam_frame_index": "image_head_frame_index",
    "image_webcam_stale": "image_head_stale",
    "image_webcam_delta_ms": "image_head_delta_ms",
    "mocap_webcam_position": "mocap_head_position",
    "mocap_webcam_orientation": "mocap_head_orientation",
    "mocap_webcam_tracked": "mocap_head_tracked",
    "mocap_webcam_rigid_markers": "mocap_head_rigid_markers",
    "mocap_webcam_transform": "mocap_head_transform",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memmap-dir", type=Path, required=True)
    ap.add_argument("--allintra-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    memmap = args.memmap_dir
    videos = args.allintra_root

    # 1) Rename .dat files + manifest entries.
    manifest_path = memmap / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    renamed_fields = []
    for old, new in FIELD_RENAMES.items():
        info = manifest["fields"].get(old)
        if info is None:
            continue
        old_file = memmap / info["filename"]
        new_file = memmap / f"{new}.dat"
        if not args.dry_run:
            if new_file.exists():
                raise FileExistsError(f"{new_file} already exists")
            old_file.rename(new_file)
        info["filename"] = f"{new}.dat"
        manifest["fields"][new] = info
        del manifest["fields"][old]
        renamed_fields.append(old)
    if not args.dry_run:
        shutil.copy2(manifest_path, manifest_path.with_suffix(".json.bak"))
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"renamed {len(renamed_fields)} frame fields: "
          f"{', '.join(renamed_fields)}")

    # 2) Metadata key + path values.
    md_path = memmap / "metadata.npz"
    md = np.load(md_path, allow_pickle=False)
    if "episode_webcam_video_path" in md:
        values = md["episode_webcam_video_path"]
        new_values = np.asarray(
            [v if (not v or b"_head.mp4" in v)
             else v.replace(b".mp4", b"_head.mp4")
             for v in values]
        )
        out = {k: v for k, v in md.items() if k != "episode_webcam_video_path"}
        out["episode_head_video_path"] = new_values
        if not args.dry_run:
            shutil.copy2(md_path, md_path.with_suffix(".npz.bak2"))
            np.savez(md_path, **out)
        print("renamed metadata key episode_webcam_video_path -> "
              "episode_head_video_path")
    else:
        print("metadata: episode_webcam_video_path already renamed")

    # 3) Rename the 63 head-view videos.
    n_videos = 0
    for old in sorted(videos.glob("episode_*_allintra.mp4")):
        if "head" in old.name or "wrist" in old.name or "zed" in old.name:
            continue  # already view-tagged (new streams)
        new = videos / old.name.replace("_allintra.mp4", "_head_allintra.mp4")
        if not args.dry_run:
            old.rename(new)
        n_videos += 1
        print(f"  {old.name} -> {new.name}")
    print(f"renamed {n_videos} videos")

    if args.dry_run:
        print("DRY RUN: no changes written")


if __name__ == "__main__":
    main()
