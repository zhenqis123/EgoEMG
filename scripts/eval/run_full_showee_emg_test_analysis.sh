#!/usr/bin/env bash
# Full-time-step EgoEMG test_analysis for the completed full-data EMGFormer S/L runs.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${RUN_ROOT:-data/logs/full_showee_s_l_all_fusions_20260721}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}/test_analysis_emg}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-128}

run_eval() {
    local name=$1
    local experiment=$2
    local checkpoint=$3
    local output_dir=${OUTPUT_ROOT}/${name}
    local hydra_checkpoint=${checkpoint//=/\\=}
    mkdir -p "$output_dir"
    (
        cd "$output_dir"
        python -m egoemg.test_analysis \
            "experiment=${experiment}" \
            "checkpoint=${hydra_checkpoint}" \
            +per_group_stats=true \
            "batch_size=${BATCH_SIZE}" \
            "hydra.run.dir=${output_dir}/hydra"
    ) 2>&1 | tee "${output_dir}/console.log"
}

cd "$REPO"
run_eval emgformer_s \
    emgformer/regression_egoemg_showee_small \
    "${RUN_ROOT}/emgformer_s/train/version_0/checkpoints/egoemg-incre-showee-small-epoch=124-val_mae=0.2490.ckpt"
run_eval emgformer_l \
    emgformer/regression_egoemg_showee_large \
    "${RUN_ROOT}/emgformer_l/train/version_0/checkpoints/egoemg-incre-showee-large-epoch=097-val_mae=0.2422.ckpt"
