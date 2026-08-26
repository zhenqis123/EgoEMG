#!/bin/bash
# Download the EgoEMG dataset package for the EgoEMG benchmark.
#
# The small preview package (dataset_egoemg_preview) contains a self-contained
# three-episode v3 shard (flat layout): memmap_data, webcam all-intra videos,
# pre-crop LMDB shards, metadata/calibration, and a README.
#
# Two mirrored sources are supported:
#   gdrive    (default) Google Drive folder, via gdown
#   baidupcs  Baidu Netdisk path /EgoEMG_release/dataset_egoemg_preview (requires login)
#
# Usage:
#   bash scripts/download/download_egoemg_data.sh [--source gdrive|baidupcs] [outdir]

set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE="gdrive"
OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"; shift 2 ;;
    *)
      OUT_DIR="$1"; shift ;;
  esac
done
OUT_DIR="${OUT_DIR:-data/dataset_egoemg_preview}"
mkdir -p "$OUT_DIR"

case "$SOURCE" in
  gdrive)
    GDRIVE_FOLDER_ID="12C6Q1CD1uihJhx4s0Rm2s7Um76Kh8rG1"
    echo "Downloading EgoEMG dataset from Google Drive ..."
    gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "$OUT_DIR"
    ;;
  baidupcs)
    echo "Downloading EgoEMG dataset from Baidu Netdisk (/EgoEMG_release/dataset_egoemg_preview) ..."
    baidupcs download /EgoEMG_release/dataset_egoemg_preview
    echo "Saved under ./download/ by default; move the package into $OUT_DIR if needed."
    ;;
  *)
    echo "Unknown source: '$SOURCE' (use 'gdrive' or 'baidupcs')" >&2
    exit 1
    ;;
esac

echo "EgoEMG dataset downloaded to $OUT_DIR"
