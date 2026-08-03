#!/bin/bash
# Download pretrained checkpoints for the EgoEMG benchmark experiments.
#
# Six checkpoints are provided:
#   emg2pose_emgformer_small | egoemg_emgformer_small | vision_resnet18 |
#   vision_vit_small | fusion_resnet_emgfusion | fusion_vit_emgfusion
#
# Two mirrored sources are supported:
#   gdrive    (default) Google Drive folder, via gdown
#   baidupcs  Baidu Netdisk path /EgoEMG_release/checkpoints (requires login)
#
# Usage:
#   bash scripts/download/download_checkpoints.sh            # GDrive
#   bash scripts/download/download_checkpoints.sh baidupcs   # Baidu Netdisk

set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE="${1:-gdrive}"
mkdir -p checkpoints

case "$SOURCE" in
  gdrive)
    GDRIVE_FOLDER_ID="1_JcHDs9uBIbFxbH0f41Sk95pCqXCcTFG"
    echo "Downloading checkpoints from Google Drive ..."
    gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O checkpoints
    ;;
  baidupcs)
    echo "Downloading checkpoints from Baidu Netdisk (/EgoEMG_release/checkpoints) ..."
    baidupcs download /EgoEMG_release/checkpoints
    echo "Saved under ./download/ by default; move the .ckpt files into checkpoints/ if needed."
    ;;
  *)
    echo "Unknown source: '$SOURCE' (use 'gdrive' or 'baidupcs')" >&2
    exit 1
    ;;
esac

echo "Done."
