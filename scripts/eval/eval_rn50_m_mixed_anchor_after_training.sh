#!/usr/bin/env bash
# Evaluate the actual best checkpoint selected by Lightning after the
# mixed-anchor training run finishes.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${REPO}/logs/20260726/fusion_rn50_m_egoemg_only_wl12000_crossattn_mixed_anchor_augbest_20e
VERSION_DIR=${RUN_ROOT}/train/train/version_0
RESULT_DIR=${REPO}/test_results/fusion/crossattn_mixed_anchor_augbest_20e_20260726
MODELS_JSON=${RESULT_DIR}/models.json
OUTPUT_JSON=${RESULT_DIR}/unified_center_eval.json

conda activate emg2pose_env
cd "$REPO"

python - "$VERSION_DIR/checkpoints/last.ckpt" "$MODELS_JSON" <<'PY'
import json
import sys
from pathlib import Path

import torch

last_checkpoint = Path(sys.argv[1]).resolve()
models_path = Path(sys.argv[2]).resolve()
checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
callbacks = checkpoint.get("callbacks", {})
best_path = None
for state in callbacks.values():
    if isinstance(state, dict) and state.get("best_model_path"):
        best_path = Path(state["best_model_path"]).resolve()
        break
if best_path is None or not best_path.is_file():
    raise RuntimeError(f"Could not resolve best checkpoint from {last_checkpoint}")

models = {
    "crossattn_mixed_anchor_augbest_20e_best": {
        "ckpt": str(best_path),
        "config_path": str(
            Path(
                "${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/20260726/"
                "fusion_rn50_m_egoemg_only_wl12000_crossattn_"
                "mixed_anchor_augbest_20e/train/hydra/hydra_configs/config.yaml"
            )
        ),
        "uses_vision": True,
    },
    "crossattn_zero_anchor_augbest_50e_ep47": {
        "ckpt": (
            "${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/20260726/"
            "fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_"
            "anchor_augbest_50e/train/train/version_0/checkpoints/"
            "rn18-s-8ch-centerfusion-epoch=047-val_mae=0.0924.ckpt"
        ),
        "config_path": (
            "${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/20260726/"
            "fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_"
            "anchor_augbest_50e/train/hydra/hydra_configs/config.yaml"
        ),
        "uses_vision": True,
    },
    "vision_rn50": {
        "ckpt": (
            "${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/fusion/vision_resnet50/"
            "version_0/checkpoints/resnet50-vision-epoch=145-val_mae=0.0923.ckpt"
        ),
        "config_path": (
            "${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/fusion/vision_resnet50/"
            "version_0/hparams.yaml"
        ),
        "uses_vision": True,
    },
}
models_path.parent.mkdir(parents=True, exist_ok=True)
models_path.write_text(json.dumps(models, indent=2) + "\n")
print(f"best checkpoint: {best_path}")
PY

CUDA_VISIBLE_DEVICES=0 python scripts/eval/unified_center_eval.py \
  --models-json "$MODELS_JSON" \
  --output "$OUTPUT_JSON" \
  --center-window-length 12000
