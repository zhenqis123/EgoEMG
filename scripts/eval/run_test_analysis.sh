#!/bin/bash
# Evaluate vision-only & fusion models on EgoEMG test splits.
#
# Usage:
#   bash scripts/eval/run_test_analysis.sh          # all models, GPU
#   bash scripts/eval/run_test_analysis.sh --cpu     # all models, CPU
#   bash scripts/eval/run_test_analysis.sh --models vit_small,vit_base  # selected models
#
# Results are saved to test_results/<model>/results.csv

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Defaults ────────────────────────────────────────────────────────────────
DEVICE="cuda:0"
SPLITS="user gesture both"
HANDS="left right"
BATCH_SIZE=160
MODELS="resnet18,resnet50,resnet152,vit_small,vit_base,vit_large"

# ── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)    DEVICE="cpu"; shift ;;
        --gpu)    DEVICE="cuda:0"; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra MODEL_LIST <<< "$MODELS"

# ── Model registry ──────────────────────────────────────────────────────────
# Format: "config|checkpoint|output_dir"
declare -A MODEL_CFG MODEL_CKPT MODEL_OUT

# vision_resnet18
MODEL_CFG[resnet18]="experiment/fusion/vision_resnet18"
MODEL_CKPT[resnet18]="logs/fusion/vision_resnet/version_9/checkpoints/resnet-vision-epoch=011-val_mae=0.1022.ckpt"
MODEL_OUT[resnet18]="test_results/vision_resnet18"

# vision_resnet50
MODEL_CFG[resnet50]="experiment/fusion/vision_resnet50"
MODEL_CKPT[resnet50]="logs/fusion/vision_resnet50/version_0/checkpoints/resnet50-vision-epoch=145-val_mae=0.0923.ckpt"
MODEL_OUT[resnet50]="test_results/vision_resnet50"

# vision_resnet152
MODEL_CFG[resnet152]="experiment/fusion/vision_resnet152"
MODEL_CKPT[resnet152]="logs/fusion/vision_resnet152/version_0/checkpoints/resnet152-vision-epoch=125-val_mae=0.0894.ckpt"
MODEL_OUT[resnet152]="test_results/vision_resnet152"

# vision_vit_small
MODEL_CFG[vit_small]="experiment/fusion/vision_vit_small"
MODEL_CKPT[vit_small]="logs/fusion/vision_vit_small/version_0/checkpoints/vit-small-epoch=179-val_mae=0.1053.ckpt"
MODEL_OUT[vit_small]="test_results/vision_vit_small"

# vision_vit_base
MODEL_CFG[vit_base]="experiment/fusion/vision_vit_base"
MODEL_CKPT[vit_base]="logs/fusion/vision_vit_base/version_0/checkpoints/vit-base-epoch=123-val_mae=0.1010.ckpt"
MODEL_OUT[vit_base]="test_results/vision_vit_base"

# vision_vit_large
MODEL_CFG[vit_large]="experiment/fusion/vision_vit_large"
MODEL_CKPT[vit_large]="logs/fusion/vision_vit_large/version_0/checkpoints/vit-large-epoch=127-val_mae=0.0940.ckpt"
MODEL_OUT[vit_large]="test_results/vision_vit_large"

# ── Run ─────────────────────────────────────────────────────────────────────
echo "=============================================="
echo "test_analysis_fusion batch evaluation"
echo "Device:      $DEVICE"
echo "Models:      ${MODEL_LIST[*]}"
echo "Splits:      $SPLITS"
echo "Hands:       $HANDS"
echo "Batch size:  $BATCH_SIZE"
echo "=============================================="

for model in "${MODEL_LIST[@]}"; do
    cfg="${MODEL_CFG[$model]:?unknown model: $model}"
    ckpt="${MODEL_CKPT[$model]}"
    out="${MODEL_OUT[$model]}"

    if [[ ! -f "$ckpt" ]]; then
        echo "SKIP $model: checkpoint not found: $ckpt"
        continue
    fi

    mkdir -p "$out"

    echo ""
    echo "===== $model ====="
    echo "  config:     $cfg"
    echo "  checkpoint: $ckpt"
    echo "  output:     $out/results.csv"
    echo ""

    python -m emg2pose.test_analysis_fusion \
        --config-name "$cfg" \
        --checkpoint "$ckpt" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --output "$out/results.csv" \
        --splits $SPLITS \
        --hands $HANDS

    echo ""
    echo "  Done: $out/results.csv"
done

echo ""
echo "All models evaluated."
