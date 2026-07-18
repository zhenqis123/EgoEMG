# egoemg_vemg2pose_wo_aug

vEMG2Pose baseline (original architecture) on EgoEMG, no augmentation.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_vemg2pose_wo_aug_mae0.2750_epoch065.ckpt`
- **Config**: `experiment=emg2pose/regression_vemg2pose_egoemg`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2604 | 0.2611 ± 0.0247 | 36 |
| user | 0.2966 | 0.2851 ± 0.0297 | 6 |
| both | 0.3059 | 0.3025 ± 0.0221 | 5 |
| **overall** | **0.2780** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2873 | 0.2734 ± 0.0276 (6) | 0.2863 ± 0.0424 (52) |
| user/right | 0.3057 | 0.2964 ± 0.0392 (6) | 0.3054 ± 0.0427 (52) |
| gesture/left | 0.2558 | 0.2572 ± 0.0288 (36) | 0.2295 ± 0.0696 (17) |
| gesture/right | 0.2651 | 0.2648 ± 0.0280 (36) | 0.2400 ± 0.0727 (17) |
| both/left | 0.3046 | 0.2995 ± 0.0258 (5) | 0.3002 ± 0.0556 (16) |
| both/right | 0.3069 | 0.3051 ± 0.0339 (5) | 0.3028 ± 0.0452 (16) |

Simple mean test_mae: **0.2876**
