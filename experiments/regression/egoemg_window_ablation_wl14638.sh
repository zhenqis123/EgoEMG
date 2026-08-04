#!/usr/bin/env bash
# EgoEMG window-ablation WL=14638 — recipe aligned to historical trial_0006.
# Reference: logs/optuna_window/egoemg-window-v4/trial_0006_2026-05-21_11-25-39
# Config:    config/experiment/emgformer/regression_egoemg_window_ablation_wl14638.yaml
#
# Layout: target_hand (raw 8 channels per hand, no 16-ch sparse-place).
# This is the "no channel interpolation" variant.
#
# Verified to reproduce test_mae ≈ 0.2424 (vs historical 0.2401) with the
# field-aware norm fix and 8-channel layout.
#
# Key values that the config captures (do NOT override these):
#   featurizer   = bespoke TdsNetwork (8ch input conv, kernel=9/3 stages,
#                  num_blocks=1; NOT tds_slim and NOT canonical tds_no_out)
#   decoder      = transformer/preset/middle_aggressive (num_layers=6,
#                  dropout=0.15; the name "middle" in the historical config is
#                  a label — the actual num_layers=6 values come from
#                  middle_aggressive.yaml)
#   head         = mlp with hidden_sizes=[512]
#   emg_field    = filtered (use data/EgoEMG_unified_memmap emg_*_filtered columns)
#   emg_layout   = target_hand (8ch, no interpolation)
#   seed=6, batch_size=194
#   loss_weights = mae:1.0 + fingertip_distance:0.01 (no landmark/fingertip)
#   batch_augmentation = full pipeline (all knobs explicitly set)
#   transforms   = minimal (ExtractField + RotationAugmentation + ToFloatTensor)
#   eval_multiprocessing_context = forkserver (historical default, not overridden)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

conda activate emg2pose_env

BASE_CONFIG="emgformer/regression_egoemg_window_ablation_wl14638"
GPUS="${GPUS:-0,1,2,3,4,5}"
WL=14638
EMG_FIELD="${EMG_FIELD:-filtered}"
BS="${BS:-194}"
NUM_WORKERS="${NUM_WORKERS:-12}"
LR="${LR:-0.0005}"
EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-6}"
BASE_LOG="logs/regression/egoemg_window_ablation_wl14638_aligned_${EMG_FIELD}_8ch"
EGOEMG_DATA="${EGOEMG_DATA:-data/EgoEMG_unified_memmap}"

echo "=== EgoEMG window-ablation WL=14638 (aligned to trial_0006) ==="
echo "Reference: egoemg-window-v4/trial_0006_2026-05-21_11-25-39"
echo "Config: ${BASE_CONFIG}"
echo "WL/stride: ${WL}/1463"
echo "EMG: ${EMG_FIELD}"
echo "Memmap: ${EGOEMG_DATA}"
echo "Batch size: ${BS}"
echo "Seed: ${SEED}"
echo "Logs: ${BASE_LOG}"
echo ""

mkdir -p "${BASE_LOG}"

CMD=(
python -m egoemg.train
  experiment=${BASE_CONFIG} \
  egoemg_unified_memmap_dir=${EGOEMG_DATA} \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  hydra.run.dir=${BASE_LOG} \
  dataset.train.0.dataset_name=egoemg_8ch \
  dataset.train.1.dataset_name=egoemg_8ch \
  dataset.val.0.dataset_name=egoemg_8ch \
  dataset.val.1.dataset_name=egoemg_8ch \
  dataset.test.0.dataset_name=egoemg_8ch \
  dataset.test.1.dataset_name=egoemg_8ch \
  batch_size=${BS} \
  num_workers=${NUM_WORKERS} \
  optimizer.lr=${LR} \
  seed=${SEED}
)

if [[ -t 1 && "${LOG_TO_FILE:-0}" != "1" ]]; then
  "${CMD[@]}"
else
  "${CMD[@]}" 2>&1 | tee "${BASE_LOG}/console.log"
fi
