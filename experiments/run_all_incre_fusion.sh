#!/usr/bin/env bash
# Sequential experiment pipeline: EMGFormer S/L cotrain + 4 fusion variants.
# Each step is independently runnable; the script runs them in order.
#
# Prerequisites:
#   - GPU 2,3,4,5 available (4 GPUs)
#   - data/EgoEMG_memmap and data/EgoEMG_incre/data_right_merged present
#   - Vision crops at data/EgoEMG_v2_crops
#   - Vision-only checkpoints at logs/fusion/vision_resnet/version_9/ and vision_vit_small/version_0/
#
# Usage:
#   bash experiments/run_all_incre_fusion.sh            # run all
#   bash experiments/run_all_incre_fusion.sh --step 1   # run only step 1
#   bash experiments/run_all_incre_fusion.sh --from 3   # run from step 3
set -euo pipefail
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

conda activate emg2pose_env

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="${GPUS:-2,3,4,5}"
EPOCHS="${EPOCHS:-150}"
TS=$(date +%Y%m%d_%H%M%S)

# Parse args
STEP=""
FROM=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --step) STEP="$2"; shift 2;;
    --from) FROM="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

should_run() {
  local n=$1
  if [ -n "$STEP" ]; then [ "$n" == "$STEP" ]; else [ "$n" -ge "$FROM" ]; fi
}

wait_gpus_free() {
  echo "[wait] Checking GPU availability..."
  while true; do
    local busy=0
    for gpu in 2 3 4 5; do
      local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null || echo 0)
      if [ "$mem" -gt 2000 ]; then busy=1; fi
    done
    if [ "$busy" -eq 0 ]; then echo "[wait] GPUs 2-5 are free."; break; fi
    echo "[wait] GPUs busy, waiting 60s..."
    sleep 60
  done
}

# ─── Step 1: EMGFormer-S + incre cotrain ─────────────────────────────────────
if should_run 1; then
  echo ""
  echo "================================================================"
  echo "[Step 1/6] EMGFormer-S + incre cotrain (150 epochs)"
  echo "================================================================"
  LOG_DIR="logs/regression/egoemg_small_incre_cotrain_${TS}"
  mkdir -p test_results/egoemg_emgformer_small_incre_cotrain/checkpoints
  python -m emg2pose.train \
    experiment=emgformer/regression_egoemg_incre_cotrain_wl12000_small \
    augmentation=batch_aug_best_v2 \
    "trainer.devices=[${GPUS}]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=${EPOCHS} \
    seed=42 \
    hydra.run.dir="${LOG_DIR}" \
    2>&1 | tee "${LOG_DIR}/console.log"
  # Save checkpoint + results
  BEST=$(ls -t "${LOG_DIR}"/regression_egoemg_incre_cotrain_wl12000_small/version_0/checkpoints/*val_mae*.ckpt 2>/dev/null | head -1)
  if [ -n "$BEST" ]; then
    cp "$BEST" test_results/egoemg_emgformer_small_incre_cotrain/checkpoints/best.ckpt
    echo "[Step 1] Saved checkpoint: test_results/egoemg_emgformer_small_incre_cotrain/checkpoints/best.ckpt"
  else
    echo "[Step 1] ERROR: No checkpoint found!"; exit 1
  fi
  echo "[Step 1] DONE"
fi

# ─── Step 2: EMGFormer-L + incre cotrain ─────────────────────────────────────
if should_run 2; then
  echo ""
  echo "================================================================"
  echo "[Step 2/6] EMGFormer-L + incre cotrain (150 epochs)"
  echo "================================================================"
  LOG_DIR="logs/regression/egoemg_large_incre_cotrain_${TS}"
  python -m emg2pose.train \
    experiment=emgformer/regression_egoemg_incre_cotrain_wl12000_large \
    augmentation=batch_aug_best_v2 \
    "trainer.devices=[${GPUS}]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=${EPOCHS} \
    seed=42 \
    hydra.run.dir="${LOG_DIR}" \
    2>&1 | tee "${LOG_DIR}/console.log"
  echo "[Step 2] DONE (L checkpoint in ${LOG_DIR})"
fi

# ─── Step 3: F-RN18+S fusion ─────────────────────────────────────────────────
if should_run 3; then
  echo ""
  echo "================================================================"
  echo "[Step 3/6] F-RN18+S: ResNet18 + EMGFormer-S fusion (center_supervised)"
  echo "================================================================"
  # Verify EMG checkpoint exists
  if [ ! -f "test_results/egoemg_emgformer_small_incre_cotrain/checkpoints/best.ckpt" ]; then
    echo "[Step 3] ERROR: S checkpoint not found! Run Step 1 first."; exit 1
  fi
  python -m emg2pose.train \
    experiment=fusion/fusion_rn18_s_center_8ch \
    +augmentation=batch_aug_best_v2 \
    "trainer.devices=[${GPUS}]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=${EPOCHS} \
    seed=42 \
    2>&1 | tee logs/fusion/rn18_s_8ch_${TS}.log
  echo "[Step 3] DONE"
fi

# ─── Step 4: F-ViT-S+S fusion ────────────────────────────────────────────────
if should_run 4; then
  echo ""
  echo "================================================================"
  echo "[Step 4/6] F-ViT-S+S: DINOv2 ViT-S + EMGFormer-S fusion"
  echo "================================================================"
  if [ ! -f "test_results/egoemg_emgformer_small_incre_cotrain/checkpoints/best.ckpt" ]; then
    echo "[Step 4] ERROR: S checkpoint not found!"; exit 1
  fi
  python -m emg2pose.train \
    experiment=fusion/fusion_vits_s_center_8ch \
    +augmentation=batch_aug_best_v2 \
    "trainer.devices=[${GPUS}]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=${EPOCHS} \
    seed=42 \
    2>&1 | tee logs/fusion/vits_s_8ch_${TS}.log
  echo "[Step 4] DONE"
fi

# ─── Step 5: F-RN18+M fusion ─────────────────────────────────────────────────
if should_run 5; then
  echo ""
  echo "================================================================"
  echo "[Step 5/6] F-RN18+M: ResNet18 + EMGFormer-M fusion (cotrain incre)"
  echo "================================================================"
  if [ ! -f "test_results/egoemg_emgformer_middle_incre_cotrain/checkpoints/best.ckpt" ]; then
    echo "[Step 5] ERROR: M checkpoint not found!"; exit 1
  fi
  python -m emg2pose.train \
    experiment=fusion/fusion_rn18_m_center_8ch \
    +augmentation=batch_aug_best_v2 \
    "trainer.devices=[${GPUS}]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=${EPOCHS} \
    seed=42 \
    2>&1 | tee logs/fusion/rn18_m_8ch_${TS}.log
  echo "[Step 5] DONE"
fi

# ─── Step 6: F-ViT-S+M fusion ────────────────────────────────────────────────
if should_run 6; then
  echo ""
  echo "================================================================"
  echo "[Step 6/6] F-ViT-S+M: DINOv2 ViT-S + EMGFormer-M fusion (cotrain incre)"
  echo "================================================================"
  if [ ! -f "test_results/egoemg_emgformer_middle_incre_cotrain/checkpoints/best.ckpt" ]; then
    echo "[Step 6] ERROR: M checkpoint not found!"; exit 1
  fi
  python -m emg2pose.train \
    experiment=fusion/fusion_vits_m_center_8ch \
    +augmentation=batch_aug_best_v2 \
    "trainer.devices=[${GPUS}]" \
    +trainer.strategy=ddp \
    trainer.max_epochs=${EPOCHS} \
    seed=42 \
    2>&1 | tee logs/fusion/vits_m_8ch_${TS}.log
  echo "[Step 6] DONE"
fi

echo ""
echo "================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================"
