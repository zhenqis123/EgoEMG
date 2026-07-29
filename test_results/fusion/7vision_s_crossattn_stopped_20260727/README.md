# Frozen-Vision Cross-Attention Fusion Safety Baseline

This directory freezes the unified center-frame evaluation obtained on
2026-07-27 before the seven-model training queue was stopped.

Evaluation protocol:

- EgoEMG validation set, both hands
- center window length: 12000 samples
- 4154 samples in total (left 2084, right 2070)
- lower MAE is better
- improvement is `(vision_only_mae - fusion_mae) / vision_only_mae`

| Fusion variant | Checkpoint epoch | Vision-only MAE | Fusion MAE | Absolute improvement | Relative improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| RN18 + EMGFormer-S | 46 | 0.10194173 | 0.10127634 | 0.00066540 | 0.6527% |
| RN50 + EMGFormer-S | 20 | 0.09204077 | 0.09047883 | 0.00156194 | 1.6970% |
| ViT-S + EMGFormer-S | 31 | 0.10511103 | 0.10467828 | 0.00043276 | 0.4117% |
| ViT-B + EMGFormer-S | 31 | 0.10087751 | 0.10011095 | 0.00076656 | 0.7599% |
| ViT-L + EMGFormer-S | 0 | 0.09399669 | 0.09366090 | 0.00033579 | 0.3572% |

The ViT-L result is only an epoch-0 checkpoint and is therefore preliminary.
RN152 and WiLoR did not produce fusion checkpoints before the queue was
stopped, so they are not included.

Files:

- `models.json`: exact fusion checkpoint and resolved-config paths
- `unified_center_eval.json`: complete fusion metrics, including per-hand and
  per-joint results
- `vision_vitl_baseline.json`: repaired/re-evaluated ViT-L vision-only result
- `comparison.csv`: compact comparison against all corresponding vision-only
  baselines

