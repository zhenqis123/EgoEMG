#!/bin/bash
# emg2pose_v3 Window Length Sweep (1000-35000, step 2000)
# Middle model, 150 epochs per trial
set -e
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

EXPERIMENT="emgformer/regression_emg2pose"
GPUS="0,1,2,3,4,5"
LR=0.0001
EPOCHS=150
SEED=42
LOG_BASE="logs/ablation/emg2pose_window_sweep"

# 批量大小自动缩放: 目标填满 ~90% 显存 (22GB/卡)
# 实测: WL=1000, BS=2250 用了 17GB/卡 → 提高1/3 → BS=3000 约 22GB/卡
# 基准: WL=1000, BS=3000 约用 22GB/卡
# 显存 ∝ WL * BS, 所以 BS = 3000 * 1000 / WL
compute_bs() {
    local wl=$1
    python3 -c "
wl = $wl
bs = int(3000 * 1000 / wl)
bs = max(100, min(4000, bs))
print(bs)
"
}

for WL in 1000 3000 5000 7000 9000 11000 13000 15000 17000 19000 21000 23000 25000 27000 29000 31000 33000 35000; do
    BS=$(compute_bs $WL)
    STRIDE=$((WL / 2))
    TRIAL_DIR="${LOG_BASE}/wl_${WL}"

    # 跳过已完成的实验
    if [ -d "${TRIAL_DIR}/regression_emg2pose/version_0/checkpoints" ]; then
        CKPT_COUNT=$(ls ${TRIAL_DIR}/regression_emg2pose/version_0/checkpoints/emg2pose-*.ckpt 2>/dev/null | wc -l)
        if [ $CKPT_COUNT -gt 0 ]; then
            echo "[$(date)] SKIP WL=${WL} (already has checkpoints)"
            continue
        fi
    fi

    echo ""
    echo "[$(date)] WL=${WL}, BS=${BS}, stride=${STRIDE}"
    echo "[$(date)] Output: ${TRIAL_DIR}"

    python -m egoemg.train \
      experiment=${EXPERIMENT} \
      'trainer.devices=['"${GPUS}"']' \
      +trainer.strategy=ddp \
      trainer.max_epochs=${EPOCHS} \
      seed=${SEED} \
      hydra.run.dir=${TRIAL_DIR} \
      datamodule.window_length=${WL} \
      datamodule.val_test_window_length=${WL} \
      datamodule.stride=${STRIDE} \
      datamodule.val_test_stride=${WL} \
      batch_size=${BS} \
      optimizer.lr=${LR}

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[$(date)] WL=${WL} FAILED (rc=${RC})"
    else
        echo "[$(date)] WL=${WL} DONE"
    fi

    echo "[$(date)] Waiting 30s for GPU memory release..."
    sleep 30
done

echo ""
echo "============================================================"
echo "[$(date)] ALL EXPERIMENTS COMPLETE"
echo "============================================================"
