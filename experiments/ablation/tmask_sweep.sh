#!/bin/bash
# 时间掩码精细扫参 — 10 实验，4 epoch，val 每 10 step
set -e
cd /home/xiziheng/develop/emg2pose

BASE_CONFIG="emgformer/regression_egoemg_with_incre"
GPUS="0,1,2,3,4,5"
WL=12000
BS=300
LR=0.0005
EPOCHS=4
SEED=50
BASE_LOG="logs/ablation/tmask_sweep"
DATA_DIR="/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap"

# Disable all aug, then enable only target
NOAUG_EXCEPT="batch_augmentation.random_gain.mask_prob=0.0 \
  batch_augmentation.mag_warping.mask_prob=0.0 \
  batch_augmentation.baseline_drift.mask_prob=0.0 \
  batch_augmentation.powerline_noise.mask_prob=0.0 \
  batch_augmentation.channel_mask.mask_prob=0.0 \
  batch_augmentation.freq_mask.num_masks=0 \
  batch_augmentation.gaussian_noise.apply_prob=0.0"

common_args() {
  echo "experiment=${BASE_CONFIG} \
    egoemg_memmap_dir=${DATA_DIR} \
    trainer.devices=[${GPUS}] \
    +trainer.strategy=ddp \
    trainer.max_epochs=${EPOCHS} \
    hydra.run.dir=${BASE_LOG}/$(printf 'exp_%02d' $1)_$2 \
    datamodule.window_length=${WL} \
    datamodule.val_test_window_length=${WL} \
    datamodule.stride=$((WL/10)) \
    datamodule.val_test_stride=${WL} \
    batch_size=${BS} \
    optimizer.lr=${LR} \
    +trainer.val_check_interval=10 \
    seed=$((SEED + $1))"
}

run_exp() {
  local id=$1; shift
  local name=$1; shift
  local aug_overrides="$@"
  local logdir="${BASE_LOG}/$(printf 'exp_%02d' $id)_${name}"
  mkdir -p "${logdir}"
  echo "=== Exp ${id}: ${name} ==="
  python -m emg2pose.train \
    $(common_args $id $name) \
    ${aug_overrides} \
    2>&1 | tee ${logdir}/console.log
  echo "=== Exp ${id} done ==="
}

echo "=== Time Mask Fine Sweep: 10 experiments, 4 epochs ==="
echo ""

# --- 0. Baseline: best from sweep (tmask_12x500) ---
run_exp 0 "tmask_12x500_baseline" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=12 \
   batch_augmentation.time_mask.max_mask_size=500"

# --- 1. Vary number of masks (fixed size=500) ---
run_exp 1 "tmask_15x500" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=15 \
   batch_augmentation.time_mask.max_mask_size=500"

run_exp 2 "tmask_18x500" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=18 \
   batch_augmentation.time_mask.max_mask_size=500"

run_exp 3 "tmask_8x500" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=8 \
   batch_augmentation.time_mask.max_mask_size=500"

# --- 2. Vary mask size (fixed masks=12) ---
run_exp 4 "tmask_12x700" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=12 \
   batch_augmentation.time_mask.max_mask_size=700"

run_exp 5 "tmask_12x1000" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=12 \
   batch_augmentation.time_mask.max_mask_size=1000"

# --- 3. Combination: more masks + larger size ---
run_exp 6 "tmask_15x700" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=15 \
   batch_augmentation.time_mask.max_mask_size=700"

# --- 4. Best combos from sweep: tmask + drift / tmask + noise ---
run_exp 7 "tmask12_drift08" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=12 \
   batch_augmentation.time_mask.max_mask_size=500 \
   batch_augmentation.baseline_drift.mask_prob=0.8 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.01 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.10 \
   batch_augmentation.baseline_drift.min_freq=0.01 \
   batch_augmentation.baseline_drift.max_freq=0.5"

run_exp 8 "tmask12_gnoise_snr20_35" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=12 \
   batch_augmentation.time_mask.max_mask_size=500 \
   batch_augmentation.gaussian_noise.apply_prob=1.0 \
   batch_augmentation.gaussian_noise.min_snr_db=20.0 \
   batch_augmentation.gaussian_noise.max_snr_db=35.0"

# --- 5. Aggressive: very large masks ---
run_exp 9 "tmask_20x700" \
  "${NOAUG_EXCEPT} \
   batch_augmentation.time_mask.num_masks=20 \
   batch_augmentation.time_mask.max_mask_size=700"

echo ""
echo "=== All 10 experiments completed ==="
echo "Logs: ${BASE_LOG}/"
