#!/usr/bin/env bash
# One-click per-episode crop preparation for EgoEMG vision data.
#
# Usage:
#   bash scripts/prepare/run_prepare_crops.sh                              # single GPU
#   NUM_WORKERS=4 GPU_IDS="1 2" bash scripts/prepare/run_prepare_crops.sh  # multi GPU
#
# Episodes are sharded by episode_idx % num_workers == worker_id.
# Resume is automatic: episodes with .done files are skipped.
# After all episodes are prepared, a quick verification runs on episode 0.

set -euo pipefail

# Fix libffi/libstdc++ conflicts between conda base and system libraries.
export LD_PRELOAD="/lib/x86_64-linux-gnu/libffi.so.7:${CONDA_PREFIX:-$HOME/miniconda3/envs/emg2pose_env}/lib/libstdc++.so.6"

MEMMAP_DIR="${MEMMAP_DIR:-data/EgoEMG_v2_memmap}"
VIDEO_ROOT="${VIDEO_ROOT:-${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG}"
ALLINTRA_ROOT="${ALLINTRA_ROOT:-data/EgoEMG_allintra}"
OUTPUT_DIR="${OUTPUT_DIR:-data/EgoEMG_v2_crops}"

PATCH_SIZE="${PATCH_SIZE:-256}"
JPEG_QUALITY="${JPEG_QUALITY:-90}"
NUM_WORKERS="${NUM_WORKERS:-1}"
GPU_IDS="${GPU_IDS:-0}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

echo "=============================================="
echo "Per-episode crop preparation"
echo "=============================================="
echo "Memmap dir:    ${MEMMAP_DIR}"
echo "Video root:    ${VIDEO_ROOT}"
echo "Allintra root: ${ALLINTRA_ROOT}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Patch size:    ${PATCH_SIZE}"
echo "JPEG quality:  ${JPEG_QUALITY}"
echo "Num workers:   ${NUM_WORKERS}"
echo "GPU IDs:       ${GPU_IDS}"
echo "=============================================="

common_args=(
    scripts/prepare/prepare_egoemg_crops.py
    --memmap-dir "${MEMMAP_DIR}"
    --video-root "${VIDEO_ROOT}"
    --allintra-root "${ALLINTRA_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --patch-size "${PATCH_SIZE}"
    --jpeg-quality "${JPEG_QUALITY}"
    --progress
)
if [[ "${OVERWRITE}" == "1" ]]; then
    common_args+=(--overwrite)
fi

# ---- Step 1: Prepare all episodes ----
if [[ "${NUM_WORKERS}" -le 1 ]]; then
    read -ra GPU_ARRAY <<< "${GPU_IDS}"
    python "${common_args[@]}" --gpu-id "${GPU_ARRAY[0]}"
else
    read -ra GPU_ARRAY <<< "${GPU_IDS}"
    num_gpus=${#GPU_ARRAY[@]}

    echo ""
    echo "Launching ${NUM_WORKERS} workers on ${num_gpus} GPUs ..."
    pids=()
    logs=()
    for wid in $(seq 0 $((NUM_WORKERS - 1))); do
        gpu_id=${GPU_ARRAY[$((wid % num_gpus))]}
        log_file="/tmp/prepare_crops_worker${wid}.log"
        echo "  [worker ${wid}] GPU=${gpu_id} log=${log_file}"
        python "${common_args[@]}" \
            --num-workers "${NUM_WORKERS}" \
            --worker-id "${wid}" \
            --gpu-id "${gpu_id}" \
            > "${log_file}" 2>&1 &
        pids+=($!)
        logs+=("${log_file}")
    done

    echo ""
    echo "Monitor: tail -f /tmp/prepare_crops_worker*.log"
    echo ""

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
fi

# ---- Step 2: Quick verification ----
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    echo ""
    echo "=============================================="
    echo "Verifying: episode 0 LMDB"
    echo "=============================================="
    python -c "
import json, lmdb
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
manifest = json.loads((output_dir / 'manifest.json').read_text())
ep_ids = manifest.get('episode_ids', [])
print(f'Manifest: {len(ep_ids)} episodes, patch_size={manifest.get(\"patch_size\")}')

done_count = sum(1 for eid in ep_ids if (output_dir / f'{eid}.done').exists())
print(f'Completed: {done_count}/{len(ep_ids)} episodes')

if ep_ids:
    ep0 = ep_ids[0]
    lmdb_path = output_dir / f'{ep0}.lmdb'
    if lmdb_path.exists():
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with env.begin() as txn:
            n = txn.stat()['entries']
        env.close()
        done_info = json.loads((output_dir / f'{ep0}.done').read_text())
        print(f'{ep0}: {n} LMDB entries, done_info={done_info}')
    else:
        print(f'{ep0}: LMDB not found')
print('Verification OK')
"
fi

echo ""
echo "=============================================="
echo "Done. Per-episode crops saved to: ${OUTPUT_DIR}"
echo ""
echo "Train with:"
echo "  python -m emg2pose.train_vision \\"
echo "    data_location=${MEMMAP_DIR} \\"
echo "    video_root=${VIDEO_ROOT} \\"
echo "    allintra_root=${ALLINTRA_ROOT} \\"
echo "    vision_index_dir=${MEMMAP_DIR}/vision_index \\"
echo "    per_episode_crops_dir=${OUTPUT_DIR} \\"
echo "    train=True eval=True"
echo "=============================================="
