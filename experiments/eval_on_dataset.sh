#!/bin/bash
# Evaluate a checkpoint on an arbitrary memmap dataset.
#
# Usage:
#   bash experiments/eval_on_dataset.sh \
#     /path/to/checkpoint.ckpt \
#     /path/to/memmap_dir
#
#   bash experiments/eval_on_dataset.sh \
#     /path/to/checkpoint.ckpt \
#     /path/to/memmap_dir \
#     --target-hand left \
#     --ignore-head-tail 0
#
# Options:
#   --target-hand      left | right | both (default: right)
#   --ignore-head-tail  how many tail dims to strip from MAE (default: 2)
#   --dataset-name      norm-stats key (default: egoemg)
#   --allowed-splits    comma-separated split names (default: test)
#   --batch-size        batch size (default: 64)
#   --gpu               GPU index (default: 0)

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint> <memmap_dir> [options]"
    echo ""
    echo "Options:"
    echo "  --target-hand      left | right | both (default: right)"
    echo "  --ignore-head-tail  tail dims to strip (default: 2)"
    echo "  --dataset-name      norm-stats key (default: egoemg)"
    echo "  --allowed-splits    comma-separated splits (default: test)"
    echo "  --batch-size        batch size (default: 64)"
    echo "  --gpu               GPU index (default: 0)"
    exit 1
fi

CKPT="$1"
MEMMAP_DIR="$2"
shift 2

# ── Defaults ──
TARGET_HAND="right"
IGNORE_HEAD_TAIL=2
DATASET_NAME="egoemg"
ALLOWED_SPLITS="test"
BATCH_SIZE=64
GPU=0

# ── Parse options ──
while [ $# -gt 0 ]; do
    case "$1" in
        --target-hand)      TARGET_HAND="$2"; shift 2 ;;
        --ignore-head-tail) IGNORE_HEAD_TAIL="$2"; shift 2 ;;
        --dataset-name)     DATASET_NAME="$2"; shift 2 ;;
        --allowed-splits)   ALLOWED_SPLITS="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --gpu)              GPU="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Convert comma-separated splits to Hydra list ──
IFS=',' read -ra SPLIT_ARRAY <<< "$ALLOWED_SPLITS"
SPLIT_OVERRIDE="[$(printf "'%s'," "${SPLIT_ARRAY[@]}" | sed 's/,$//')]"

cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

echo "========================================"
echo "  Evaluate Checkpoint on Dataset"
echo "========================================"
echo "Checkpoint:       $CKPT"
echo "Memmap dir:       $MEMMAP_DIR"
echo "Target hand:      $TARGET_HAND"
echo "Ignore head/tail: $IGNORE_HEAD_TAIL"
echo "Dataset name:     $DATASET_NAME"
echo "Allowed splits:   $ALLOWED_SPLITS"
echo "Batch size:       $BATCH_SIZE"
echo "GPU:              $GPU"
echo ""

# ── Checks ──
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
if [ ! -d "$MEMMAP_DIR" ]; then
    echo "ERROR: Memmap dir not found: $MEMMAP_DIR"
    exit 1
fi

# ── Run ──
conda activate emg2pose_env

python -u -m emg2pose.train \
    train=False eval=True \
    experiment=emgformer/eval_on_dataset \
    "checkpoint=$CKPT" \
    "eval_memmap_dir=$MEMMAP_DIR" \
    "eval_target_hand=$TARGET_HAND" \
    "ignore_head_tail_dims=$IGNORE_HEAD_TAIL" \
    "eval_dataset_name=$DATASET_NAME" \
    "eval_allowed_splits=$SPLIT_OVERRIDE" \
    "trainer.devices=[$GPU]" \
    "batch_size=$BATCH_SIZE"
