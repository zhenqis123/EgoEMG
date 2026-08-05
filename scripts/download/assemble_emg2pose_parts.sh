#!/bin/bash
# Reassemble the split memmap files of dataset_emg2pose_benchmark.
#
# Baidu Netdisk caps single files at 128 GB, so `emg.dat` (~181 GB) and
# `joint_angles.dat` (~227 GB) are distributed as 100 GB `*.part_XX` chunks
# under /EgoEMG_release/dataset_emg2pose_benchmark/ on Baidu Netdisk. After
# downloading the directory, run this script inside it to restore the
# original files (identical to the Google Drive copies).
#
# Usage:
#   bash scripts/download/assemble_emg2pose_parts.sh <memmap_dir>
set -euo pipefail
cd "$(dirname "$0")/../.."

MM="${1:?usage: assemble_emg2pose_parts.sh <memmap_dir>}"
for f in emg.dat joint_angles.dat; do
  if [ -f "$MM/$f" ]; then
    echo "$MM/$f already exists; skipping"
    continue
  fi
  parts=$(ls "$MM"/"${f}.part_"* 2>/dev/null || true)
  if [ -z "$parts" ]; then
    echo "error: no parts found for $f in $MM" >&2
    exit 1
  fi
  # shellcheck disable=SC2086 # intentional word-splitting over part files
  cat $parts > "$MM/$f"
  echo "reassembled $MM/$f ($(stat -c%s "$MM/$f") bytes)"
done
echo "done."
