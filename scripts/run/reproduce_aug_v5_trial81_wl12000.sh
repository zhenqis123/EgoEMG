#!/usr/bin/env bash
set -euo pipefail

cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -u -m egoemg.train \
  experiment=emgformer/_archive/regression_emgformer_middle_aug_search_egoemg \
  egoemg_memmap_dir=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_memmap \
  +egoemg_emg_field_preference=filtered_paper \
  module.featurizer.conv_blocks.0.in_channels=16 \
  'trainer.devices=[1,2,3,4,5]' \
  +trainer.strategy=ddp \
  trainer.max_epochs=150 \
  +trainer.accumulate_grad_batches=2 \
  batch_size=250 \
  seed=81 \
  datamodule.window_length=12000 \
  datamodule.val_test_window_length=12000 \
  datamodule.stride=1200 \
  datamodule.val_test_stride=12000 \
  batch_augmentation.random_gain.min_gain=0.5291 \
  batch_augmentation.random_gain.max_gain=0.6739 \
  batch_augmentation.random_gain.mask_prob=0.1422 \
  batch_augmentation.mag_warping.sigma=0.1770 \
  batch_augmentation.mag_warping.num_knots=15 \
  batch_augmentation.mag_warping.mask_prob=0.0253 \
  batch_augmentation.baseline_drift.mask_prob=0.3544 \
  batch_augmentation.baseline_drift.min_freq=0.0114 \
  batch_augmentation.baseline_drift.max_freq=0.4625 \
  batch_augmentation.baseline_drift.min_amp_ratio=0.0154 \
  batch_augmentation.baseline_drift.max_amp_ratio=0.0572 \
  batch_augmentation.powerline_noise.mask_prob=0.0800 \
  batch_augmentation.powerline_noise.min_amp_ratio=0.0141 \
  batch_augmentation.powerline_noise.max_amp_ratio=0.0628 \
  batch_augmentation.powerline_noise.max_harmonic=5 \
  batch_augmentation.channel_mask.mask_prob=0.0112 \
  batch_augmentation.time_mask.num_masks=6 \
  batch_augmentation.freq_mask.num_masks=4 \
  batch_augmentation.gaussian_noise.min_snr_db=44.0937 \
  batch_augmentation.gaussian_noise.max_snr_db=50.0000 \
  batch_augmentation.gaussian_noise.apply_prob=0.7761 \
  hydra.run.dir=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/reproduce_aug_v5_trial81_wl12000/run_2026-06-17_bs250_acc2
