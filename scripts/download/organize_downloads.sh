#!/bin/bash
# Place data packages downloaded from the distribution share into the
# canonical local layout used by the configs and the README.
#
# The share (/EgoEMG_release/) names its folders differently from the local
# layout; this script owns that mapping so nothing else has to. Idempotent:
# packages whose target already exists are skipped.
#
# Usage:
#   bash scripts/download/organize_downloads.sh <downloaded_share_root> <data_root>
#
# Example (after downloading the share folders into ~/downloads):
#   bash scripts/download/organize_downloads.sh ~/downloads "$EGOEMG_ROOT/data"
set -euo pipefail

SRC="${1:?usage: organize_downloads.sh <downloaded_share_root> <data_root>}"
DST="${2:?usage: organize_downloads.sh <downloaded_share_root> <data_root>}"

MAPPINGS=(
  "dataset_egoemg_unified:EgoEMG_full_memmap"
  "dataset_emg2pose_benchmark:emg2pose_memmap"
  "dataset_egoemg_videos:EgoEMG_videos"
  "dataset_egoemg_crops:EgoEMG_crops"
)

placed=0
for m in "${MAPPINGS[@]}"; do
  name="${m%%:*}"
  rel="${m##*:}"
  src="$SRC/$name"
  dst="$DST/$rel"
  if [ ! -d "$src" ]; then
    continue
  fi
  if [ -e "$dst" ]; then
    echo "skip  $name -> $rel (target exists)"
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  mv "$src" "$dst"
  echo "placed $name -> $rel"
  placed=$((placed + 1))
done

echo "done ($placed package(s) placed)"
if [ "$placed" -eq 0 ]; then
  echo "note: no share packages found under $SRC (expected folder names:" \
       "${MAPPINGS[0]%%:*}, ${MAPPINGS[1]%%:*}, ${MAPPINGS[2]%%:*}, ${MAPPINGS[3]%%:*})" >&2
fi
