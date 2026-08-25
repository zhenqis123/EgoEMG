#!/bin/bash
# 30 实验增强扫参 — 每个 2 epoch，探索最佳增强策略
set -e
cd ${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

BASE_CONFIG="emgformer/regression_egoemg_with_incre"
GPUS="0,1,2,3,4,5"
WL=12000
BS=300
LR=0.0005
EPOCHS=2
BASE_SEED=42
BASE_LOG="logs/ablation/aug_sweep"
DATA_DIR="${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_full_memmap"

# Common Hydra args for all experiments
common_args() {
  echo "experiment=${BASE_CONFIG} \
    egoemg_unified_memmap_dir=${DATA_DIR} \
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
    +trainer.val_check_interval=1 \
    seed=$((BASE_SEED + $1))"
}

# Augmentation overrides are built by the caller per experiment
run_exp() {
  local id=$1; shift
  local name=$1; shift
  local aug_overrides="$@"
  local logdir="${BASE_LOG}/$(printf 'exp_%02d' $id)_${name}"
  mkdir -p "${logdir}"
  echo "=== Exp ${id}: ${name} ==="
  python -m egoemg.train \
    $(common_args $id $name) \
    ${aug_overrides} \
    2>&1 | tee ${logdir}/console.log
  echo "=== Exp ${id} done ==="
}

# Disable all batch augmentations
NOAUG="batch_augmentation=null"

# Disable individual augmentations (use with defaults as base)
DISABLE_GAIN="batch_augmentation.random_gain.mask_prob=0.0"
DISABLE_WARP="batch_augmentation.mag_warping.mask_prob=0.0"
DISABLE_DRIFT="batch_augmentation.baseline_drift.mask_prob=0.0"
DISABLE_POWERLINE="batch_augmentation.powerline_noise.mask_prob=0.0"
DISABLE_CHMASK="batch_augmentation.channel_mask.mask_prob=0.0"
DISABLE_TMASK="batch_augmentation.time_mask.num_masks=0"
DISABLE_FMASK="batch_augmentation.freq_mask.num_masks=0"
DISABLE_GNOISE="batch_augmentation.gaussian_noise.apply_prob=0.0"

# Disable all except the one being tested
DISABLE_ALL_EXCEPT_GAIN="${DISABLE_WARP} ${DISABLE_DRIFT} ${DISABLE_POWERLINE} ${DISABLE_CHMASK} ${DISABLE_TMASK} ${DISABLE_FMASK} ${DISABLE_GNOISE}"
DISABLE_ALL_EXCEPT_WARP="${DISABLE_GAIN} ${DISABLE_DRIFT} ${DISABLE_POWERLINE} ${DISABLE_CHMASK} ${DISABLE_TMASK} ${DISABLE_FMASK} ${DISABLE_GNOISE}"
DISABLE_ALL_EXCEPT_DRIFT="${DISABLE_GAIN} ${DISABLE_WARP} ${DISABLE_POWERLINE} ${DISABLE_CHMASK} ${DISABLE_TMASK} ${DISABLE_FMASK} ${DISABLE_GNOISE}"
DISABLE_ALL_EXCEPT_POWERLINE="${DISABLE_GAIN} ${DISABLE_WARP} ${DISABLE_DRIFT} ${DISABLE_CHMASK} ${DISABLE_TMASK} ${DISABLE_FMASK} ${DISABLE_GNOISE}"
DISABLE_ALL_EXCEPT_CHMASK="${DISABLE_GAIN} ${DISABLE_WARP} ${DISABLE_DRIFT} ${DISABLE_POWERLINE} ${DISABLE_TMASK} ${DISABLE_FMASK} ${DISABLE_GNOISE}"
DISABLE_ALL_EXCEPT_TMASK="${DISABLE_GAIN} ${DISABLE_WARP} ${DISABLE_DRIFT} ${DISABLE_POWERLINE} ${DISABLE_CHMASK} ${DISABLE_FMASK} ${DISABLE_GNOISE}"
DISABLE_ALL_EXCEPT_FMASK="${DISABLE_GAIN} ${DISABLE_WARP} ${DISABLE_DRIFT} ${DISABLE_POWERLINE} ${DISABLE_CHMASK} ${DISABLE_TMASK} ${DISABLE_GNOISE}"
DISABLE_ALL_EXCEPT_GNOISE="${DISABLE_GAIN} ${DISABLE_WARP} ${DISABLE_DRIFT} ${DISABLE_POWERLINE} ${DISABLE_CHMASK} ${DISABLE_TMASK} ${DISABLE_FMASK}"

# === 30 Experiments ===

# --- 0. Controls (2) ---
run_exp 0 "no_aug" \
  "${NOAUG}"

run_exp 1 "default" \
  ""

# --- 1. random_gain sweep (4) — only gain active ---
run_exp 2 "gain_mask03" \
  "${DISABLE_ALL_EXCEPT_GAIN} \
   batch_augmentation.random_gain.mask_prob=0.3"

run_exp 3 "gain_mask05" \
  "${DISABLE_ALL_EXCEPT_GAIN} \
   batch_augmentation.random_gain.mask_prob=0.5"

run_exp 4 "gain_mask07" \
  "${DISABLE_ALL_EXCEPT_GAIN} \
   batch_augmentation.random_gain.mask_prob=0.7"

run_exp 5 "gain_wide" \
  "${DISABLE_ALL_EXCEPT_GAIN} \
   batch_augmentation.random_gain.mask_prob=0.5 \
   batch_augmentation.random_gain.min_gain=0.2 \
   batch_augmentation.random_gain.max_gain=0.9"

# --- 2. gaussian_noise sweep (5) — only noise active ---
run_exp 6 "gnoise_snr50_60" \
  "${DISABLE_ALL_EXCEPT_GNOISE} \
   batch_augmentation.gaussian_noise.apply_prob=1.0 \
   batch_augmentation.gaussian_noise.min_snr_db=50.0 \
   batch_augmentation.gaussian_noise.max_snr_db=60.0"

run_exp 7 "gnoise_snr35_45" \
  "${DISABLE_ALL_EXCEPT_GNOISE} \
   batch_augmentation.gaussian_noise.apply_prob=1.0 \
   batch_augmentation.gaussian_noise.min_snr_db=35.0 \
   batch_augmentation.gaussian_noise.max_snr_db=45.0"

run_exp 8 "gnoise_snr20_35" \
  "${DISABLE_ALL_EXCEPT_GNOISE} \
   batch_augmentation.gaussian_noise.apply_prob=1.0 \
   batch_augmentation.gaussian_noise.min_snr_db=20.0 \
   batch_augmentation.gaussian_noise.max_snr_db=35.0"

run_exp 9 "gnoise_snr10_25" \
  "${DISABLE_ALL_EXCEPT_GNOISE} \
   batch_augmentation.gaussian_noise.apply_prob=1.0 \
   batch_augmentation.gaussian_noise.min_snr_db=10.0 \
   batch_augmentation.gaussian_noise.max_snr_db=25.0"

run_exp 10 "gnoise_snr5_15" \
  "${DISABLE_ALL_EXCEPT_GNOISE} \
   batch_augmentation.gaussian_noise.apply_prob=1.0 \
   batch_augmentation.gaussian_noise.min_snr_db=5.0 \
   batch_augmentation.gaussian_noise.max_snr_db=15.0"

# --- 3. time_mask sweep (4) ---
run_exp 11 "tmask_3x500" \
  "${DISABLE_ALL_EXCEPT_TMASK} \
   batch_augmentation.time_mask.num_masks=3 \
   batch_augmentation.time_mask.max_mask_size=500"

run_exp 12 "tmask_9x500" \
  "${DISABLE_ALL_EXCEPT_TMASK} \
   batch_augmentation.time_mask.num_masks=9 \
   batch_augmentation.time_mask.max_mask_size=500"

run_exp 13 "tmask_6x1000" \
  "${DISABLE_ALL_EXCEPT_TMASK} \
   batch_augmentation.time_mask.num_masks=6 \
   batch_augmentation.time_mask.max_mask_size=1000"

run_exp 14 "tmask_12x500" \
  "${DISABLE_ALL_EXCEPT_TMASK} \
   batch_augmentation.time_mask.num_masks=12 \
   batch_augmentation.time_mask.max_mask_size=500"

# --- 4. freq_mask sweep (3) ---
run_exp 15 "fmask_2x128" \
  "${DISABLE_ALL_EXCEPT_FMASK} \
   batch_augmentation.freq_mask.num_masks=2 \
   batch_augmentation.freq_mask.max_mask_size=128"

run_exp 16 "fmask_8x128" \
  "${DISABLE_ALL_EXCEPT_FMASK} \
   batch_augmentation.freq_mask.num_masks=8 \
   batch_augmentation.freq_mask.max_mask_size=128"

run_exp 17 "fmask_4x256" \
  "${DISABLE_ALL_EXCEPT_FMASK} \
   batch_augmentation.freq_mask.num_masks=4 \
   batch_augmentation.freq_mask.max_mask_size=256"

# --- 5. baseline_drift sweep (4) ---
run_exp 18 "drift_amp003" \
  "${DISABLE_ALL_EXCEPT_DRIFT} \
   batch_augmentation.baseline_drift.mask_prob=0.5 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.005 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.03"

run_exp 19 "drift_amp010" \
  "${DISABLE_ALL_EXCEPT_DRIFT} \
   batch_augmentation.baseline_drift.mask_prob=0.5 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.02 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.10"

run_exp 20 "drift_amp020" \
  "${DISABLE_ALL_EXCEPT_DRIFT} \
   batch_augmentation.baseline_drift.mask_prob=0.5 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.05 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.20"

run_exp 21 "drift_mask08" \
  "${DISABLE_ALL_EXCEPT_DRIFT} \
   batch_augmentation.baseline_drift.mask_prob=0.8 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.01 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.10"

# --- 6. channel_mask sweep (2) ---
run_exp 22 "chmask_prob01" \
  "${DISABLE_ALL_EXCEPT_CHMASK} \
   batch_augmentation.channel_mask.mask_prob=0.1"

run_exp 23 "chmask_prob03" \
  "${DISABLE_ALL_EXCEPT_CHMASK} \
   batch_augmentation.channel_mask.mask_prob=0.3"

# --- 7. Combined augmentations (6) ---
# Light combo: gain + gnoise + tmask at moderate levels
run_exp 24 "combo_light" \
  "${DISABLE_WARP} ${DISABLE_DRIFT} ${DISABLE_POWERLINE} ${DISABLE_CHMASK} ${DISABLE_FMASK} \
   batch_augmentation.random_gain.mask_prob=0.3 \
   batch_augmentation.gaussian_noise.apply_prob=0.8 \
   batch_augmentation.gaussian_noise.min_snr_db=40.0 \
   batch_augmentation.gaussian_noise.max_snr_db=55.0 \
   batch_augmentation.time_mask.num_masks=4 \
   batch_augmentation.time_mask.max_mask_size=400"

# Heavy combo: gain + gnoise + tmask + fmask at strong levels
run_exp 25 "combo_heavy" \
  "${DISABLE_WARP} ${DISABLE_DRIFT} ${DISABLE_POWERLINE} ${DISABLE_CHMASK} \
   batch_augmentation.random_gain.mask_prob=0.6 \
   batch_augmentation.random_gain.min_gain=0.3 \
   batch_augmentation.random_gain.max_gain=0.8 \
   batch_augmentation.gaussian_noise.apply_prob=1.0 \
   batch_augmentation.gaussian_noise.min_snr_db=20.0 \
   batch_augmentation.gaussian_noise.max_snr_db=40.0 \
   batch_augmentation.time_mask.num_masks=8 \
   batch_augmentation.time_mask.max_mask_size=800 \
   batch_augmentation.freq_mask.num_masks=6 \
   batch_augmentation.freq_mask.max_mask_size=200"

# Drift + powerline + channel combo
run_exp 26 "combo_drift_powerline_chmask_light" \
  "${DISABLE_GAIN} ${DISABLE_WARP} ${DISABLE_TMASK} ${DISABLE_FMASK} ${DISABLE_GNOISE} \
   batch_augmentation.baseline_drift.mask_prob=0.4 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.01 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.06 \
   batch_augmentation.powerline_noise.mask_prob=0.2 \
   batch_augmentation.channel_mask.mask_prob=0.1"

run_exp 27 "combo_drift_powerline_chmask_heavy" \
  "${DISABLE_GAIN} ${DISABLE_WARP} ${DISABLE_TMASK} ${DISABLE_FMASK} ${DISABLE_GNOISE} \
   batch_augmentation.baseline_drift.mask_prob=0.7 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.03 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.15 \
   batch_augmentation.powerline_noise.mask_prob=0.4 \
   batch_augmentation.powerline_noise.min_amp_ratio=0.01 \
   batch_augmentation.powerline_noise.max_amp_ratio=0.10 \
   batch_augmentation.channel_mask.mask_prob=0.3"

# All-in variants
run_exp 28 "allin_light" \
  "${DISABLE_WARP} ${DISABLE_POWERLINE} \
   batch_augmentation.random_gain.mask_prob=0.25 \
   batch_augmentation.gaussian_noise.apply_prob=0.7 \
   batch_augmentation.gaussian_noise.min_snr_db=40.0 \
   batch_augmentation.gaussian_noise.max_snr_db=55.0 \
   batch_augmentation.time_mask.num_masks=3 \
   batch_augmentation.time_mask.max_mask_size=400 \
   batch_augmentation.freq_mask.num_masks=2 \
   batch_augmentation.freq_mask.max_mask_size=100 \
   batch_augmentation.baseline_drift.mask_prob=0.3 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.01 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.05 \
   batch_augmentation.channel_mask.mask_prob=0.05"

run_exp 29 "allin_heavy" \
  "${DISABLE_WARP} ${DISABLE_POWERLINE} \
   batch_augmentation.random_gain.mask_prob=0.5 \
   batch_augmentation.random_gain.min_gain=0.2 \
   batch_augmentation.random_gain.max_gain=0.85 \
   batch_augmentation.gaussian_noise.apply_prob=1.0 \
   batch_augmentation.gaussian_noise.min_snr_db=15.0 \
   batch_augmentation.gaussian_noise.max_snr_db=35.0 \
   batch_augmentation.time_mask.num_masks=10 \
   batch_augmentation.time_mask.max_mask_size=800 \
   batch_augmentation.freq_mask.num_masks=6 \
   batch_augmentation.freq_mask.max_mask_size=200 \
   batch_augmentation.baseline_drift.mask_prob=0.5 \
   batch_augmentation.baseline_drift.min_amp_ratio=0.02 \
   batch_augmentation.baseline_drift.max_amp_ratio=0.12 \
   batch_augmentation.channel_mask.mask_prob=0.2"

echo ""
echo "=== All 30 experiments completed ==="
echo "Logs: ${BASE_LOG}/"
echo ""
echo "To extract results:"
echo "  python experiments/ablation/extract_sweep_results.py"
