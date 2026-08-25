#!/usr/bin/env bash
# One-click ViT feature precomputation for EgoEMG vision data.
#
# Reads pre-cropped per-episode LMDBs, runs frozen ViT backbone, saves features
# to per-episode LMDBs with the same key scheme: {frame_idx:08d}_{L/R} -> features.
#
# Usage:
#   bash scripts/prepare/run_precompute_vit_features.sh                                  # 5 GPUs (default)
#   GPU_IDS="0 1 2 3" bash scripts/prepare/run_precompute_vit_features.sh                # 4 GPUs
#   GPU_IDS="0,1,2,3" bash scripts/prepare/run_precompute_vit_features.sh                # 4 GPUs (comma)
#
# Episodes are split evenly across GPUs — one worker per GPU, no overlap.

set -euo pipefail

export LD_PRELOAD="/lib/x86_64-linux-gnu/libffi.so.7:${CONDA_PREFIX:-$HOME/miniconda3/envs/emg2pose_env}/lib/libstdc++.so.6"

# ── Configurable via env vars ─────────────────────────────────────────────
CROPS_DIR="${CROPS_DIR:-data/EgoEMG_crops}"
OUTPUT_DIR="${OUTPUT_DIR:-data/EgoEMG_v2_vit_features_lmdb}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${WILOR_PATH:-../WiLoR}/pretrained_models/wilor_final.ckpt}"

BATCH_SIZE="${BATCH_SIZE:-780}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_IDS="${GPU_IDS:-1 2 3 4 5}"

# ── Print config ──────────────────────────────────────────────────────────
echo "=============================================="
echo "Precompute ViT features (per-episode LMDBs)"
echo "=============================================="
echo "Crops dir:        ${CROPS_DIR}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Pretrained path:  ${PRETRAINED_PATH}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Num workers:      ${NUM_WORKERS}"
echo "GPU IDs:          ${GPU_IDS}"
echo "=============================================="

# ── Get total episode count from crops .done files ───────────────────────
N_EPISODES=$(find "${CROPS_DIR}" -maxdepth 1 -name '*.done' | wc -l)
echo "Total episodes:   ${N_EPISODES}"

# ── Parse GPU IDs (support both comma and space separated) ────────────────
GPU_IDS_SPACE=$(echo "${GPU_IDS}" | tr ',' ' ')
read -ra GPU_ARRAY <<< "${GPU_IDS_SPACE}"
num_gpus=${#GPU_ARRAY[@]}

# One worker per GPU, split episodes evenly.
total_workers=${num_gpus}
ep_per_gpu=$(( (N_EPISODES + num_gpus - 1) / num_gpus ))

echo "Launching ${total_workers} workers (1 per GPU), ~${ep_per_gpu} episodes each ..."
pids=()
logs=()
for wid in $(seq 0 $((total_workers - 1))); do
    ep_start=$(( wid * ep_per_gpu ))
    ep_end=$(( ep_start + ep_per_gpu ))
    if (( ep_start >= N_EPISODES )); then
        break
    fi
    if (( ep_end > N_EPISODES )); then
        ep_end=${N_EPISODES}
    fi

    gpu_id=${GPU_ARRAY[$wid]}
    log_file="/tmp/precompute_vit_worker${wid}.log"

    echo "  [worker ${wid}] GPU=${gpu_id} episodes=[${ep_start}, ${ep_end}) log=${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/prepare/precompute_vit_features_to_lmdb.py \
        --crops-dir "${CROPS_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --pretrained-path "${PRETRAINED_PATH}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --episode-start "${ep_start}" \
        --episode-end "${ep_end}" \
        --device cuda:0 \
        > "${log_file}" 2>&1 &

    pids+=($!)
    logs+=("${log_file}")
done

echo ""
echo "Monitor: tail -f /tmp/precompute_vit_worker*.log"
echo ""

# ── Wait for all workers ─────────────────────────────────────────────────
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ERROR: worker ${i} failed! See ${logs[$i]}" >&2
        fail=1
    else
        echo "  [worker ${i}] done"
    fi
done
if [[ "${fail}" -eq 1 ]]; then
    echo "ERROR: one or more workers failed." >&2
    exit 1
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="
DONE_COUNT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name '*.done' | wc -l)
echo "Episodes with features: ${DONE_COUNT} / ${N_EPISODES}"
echo ""
echo "Train with:"
echo "  python -m egoemg.train_vision \\"
echo "    data_location=<memmap_dir> \\"
echo "    cached_vit_features_dir=${OUTPUT_DIR} \\"
echo "    supervision_target=vision_only_angle \\"
echo "    train=True eval=True"
echo "=============================================="
