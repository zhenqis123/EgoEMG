#!/usr/bin/env bash
# Reproduce best V2 augmentation config (aug-v2 sweep trial #12, val_mae=0.2446).
#
# Fixed: WL=12000, EMGFormer-Middle, per-channel norm, target_hand 8ch.
# Changed: augmentation = batch_aug_best_v2 (MixUp + stronger drift + more freq masks).
set -euo pipefail
cd ${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

conda activate egoemg_env

GPUS="${GPUS:-2,3,4,5}"
EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-42}"
BASE_LOG="logs/regression/egoemg_best_v2_aug"

echo "=== EgoEMG Best V2 Augmentation (trial 12 config) ==="
echo "Augmentation: batch_aug_best_v2 (val_mae=0.2446, -1.4% vs baseline)"
echo "GPU: ${GPUS} | Epochs: ${EPOCHS} | Seed: ${SEED}"
echo ""

python -m egoemg.train \
  experiment=emgformer/regression_egoemg_window_ablation_wl12000 \
  egoemg_unified_memmap_dir=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_full_memmap \
  "trainer.devices=[${GPUS}]" \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  seed=${SEED} \
  hydra.run.dir=${BASE_LOG} \
  datamodule.window_length=12000 \
  datamodule.val_test_window_length=12000 \
  datamodule.stride=1200 \
  datamodule.val_test_stride=12000 \
  egoemg_emg_layout=target_hand \
  +egoemg_emg2pose_channel_indices=null \
  datamodule.per_dataset_norm_stats_path=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json \
  augmentation=batch_aug_best_v2 \
  2>&1 | tee ${BASE_LOG}/console.log
