#!/usr/bin/env bash
set -euo pipefail

# Launch EMG2Pose Incre Small Training (ONLY Ep 4,5)
# Uses: EgoEMG full + Incre train (Ep 4-5 only) -> Incre val eval
# Model: Small (256d, 3 layers, 512 ffn)
# GPUs: 0,1,2,3,4,5 (6x)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

# Activate conda
conda activate emg2pose_env

echo "=== Launching EMG2Pose Incre Small Training (ONLY Ep 4,5) ==="
echo "Date: $(date)"
echo "Project: $PROJECT_DIR"

python -m emg2pose.train \
  experiment=emgformer/regression_egoemg_incre_small_only45 \
  'trainer.devices=[0,1,2,3,4,5]' \
  +trainer.strategy=ddp \
  batch_size=360

echo "=== Training finished at $(date) ==="
