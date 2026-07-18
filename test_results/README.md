# `test_results/` organization

- **Per-experiment folders**: each has `results.csv`, `README.md`, and `checkpoints/`.
- **[all_results_index.csv](all_results_index.csv)**: one-line index of every experiment.
- **`references/`**: external or comparison baselines.
- **`summaries/`**: human-written multi-run summaries.

## How to read the index

- **emg2pose** experiments: primary metric is `test_mae` on the `user_stage` split.
- **EgoEMG** experiments: primary metric is the sample-weighted mean of `test_mae` over all 6 test splits.
- **vision** experiments: same as EgoEMG (center-frame evaluation on 6 splits, sample-weighted mean).
- **fusion** experiment: `center_mae` metric.
- Per-user and per-gesture stats (mean ± std) are included where available.

## Evaluation convention

Reuse the training experiment yaml and override runtime fields: `checkpoint=...`, `+per_user=true`, `+per_group_stats=true`, `hydra.run.dir=...`.

## EMG2Pose experiments (EMGFormer on EMG2Pose v3 dataset)

All aggressive configs evaluated with `norm_mode=per-dataset` (matching training configs for middle/large; small was trained with `null` but evaluated with `per-dataset` for fair comparison).
Re-evaluated 2026-05-03 with per-user stats.

| Experiment | test_mae (user_stage) | per-user MAE ± std |
|---|---|---|---|
| `emg2pose_emgformer_small_aggressive` | 0.215 | 0.217 ± 0.019 (n=20) |
| `emg2pose_emgformer_middle_aggressive` | 0.215 | 0.216 ± 0.019 (n=20) |
| `emg2pose_emgformer_large_aggressive` | 0.213 | 0.214 ± 0.019 (n=20) |
| `emg2pose_small_default` | 0.218 | — |
| `emg2pose_middle_default` | 0.212 | — |
| `emg2pose_small_ultra` | 0.212 | — |

## EgoEMG experiments

### EMGFormer (with augmentation, pretrained checkpoint)
Re-evaluated 2026-05-03 with pooled left+right per-user stats.

| Experiment | overall test_mae | pooled per-user (gesture / user / both) |
|---|---|---|---|
| `egoemg_emgformer_small_with_aug` | 0.262 | 0.246±0.026 (n=36) / 0.260±0.046 (n=6) / 0.284±0.018 (n=5) |
| `egoemg_emgformer_middle_with_aug` | 0.263 | 0.234±0.023 (n=36) / 0.274±0.038 (n=6) / 0.306±0.023 (n=5) |
| `egoemg_emgformer_large_with_aug` | 0.262 | 0.233±0.024 (n=36) / 0.272±0.045 (n=6) / 0.307±0.021 (n=5) |

### EMGFormer (no augmentation, from scratch)
Re-evaluated 2026-05-05.

| Experiment | overall test_mae | pooled per-user (gesture / user / both) |
|---|---|---|---|
| `egoemg_emgformer_small_wo_aug` | 0.2562 | 0.223±0.025 (n=36) / 0.273±0.043 (n=6) / 0.304±0.021 (n=5) |
| `egoemg_emgformer_middle_wo_aug` | 0.2482 | 0.206±0.028 (n=36) / 0.285±0.025 (n=6) / 0.303±0.016 (n=5) |
| `egoemg_emgformer_large_wo_aug` | 0.2481 | 0.204±0.026 (n=36) / 0.273±0.053 (n=6) / 0.308±0.019 (n=5) |

### Traditional architectures on EgoEMG
Re-evaluated 2026-05-05. All trained from scratch.

| Experiment | overall test_mae | pooled per-user (gesture / user / both) |
|---|---|---|---|
| `egoemg_vemg2pose_with_aug` | 0.2743 | 0.267±0.026 (n=36) / 0.260±0.049 (n=6) / 0.280±0.018 (n=5) |
| `egoemg_vemg2pose_wo_aug` | 0.2780 | 0.261±0.025 (n=36) / 0.285±0.030 (n=6) / 0.302±0.022 (n=5) |
| `egoemg_emg2pose_with_aug` | 0.2733 | 0.260±0.024 (n=36) / 0.272±0.032 (n=6) / 0.294±0.017 (n=5) |
| `egoemg_emg2pose_wo_aug` | 0.2753 | 0.270±0.023 (n=36) / 0.258±0.051 (n=6) / 0.284±0.010 (n=5) |
| `egoemg_neuropose_with_aug` | 0.2786 | 0.276±0.020 (n=36) / 0.264±0.034 (n=6) / 0.278±0.015 (n=5) |
| `egoemg_neuropose_wo_aug` | 0.2809 | 0.276±0.021 (n=36) / 0.274±0.023 (n=6) / 0.285±0.012 (n=5) |

## Vision-only experiments

Center-frame evaluation on EgoEMG test splits.

| Experiment | overall test_mae |
|---|---|
| `vision_resnet18` | 0.102 |
| `vision_resnet50` | 0.092 |
| `vision_resnet152` | 0.089 |
| `vision_vit_small` | 0.105 |
| `vision_vit_base` | 0.101 |
| `vision_vit_large` | 0.094 |

## Fusion experiment

| Experiment | center_mae |
|---|---|
| `fusion_resnet_small_emgfusion_center` | 0.100 |

## Notes

- EMG2Pose and EgoEMG numbers are not directly comparable (different datasets, splits, normalization).
- All aggressive EMG2Pose configs use `norm_mode=per-dataset` for evaluation, matching the training configs and the paper results.
- The small aggressive config incorrectly used `norm_mode=null` during training but was evaluated with `per-dataset` for fair comparison; its training config should be fixed for future runs.
