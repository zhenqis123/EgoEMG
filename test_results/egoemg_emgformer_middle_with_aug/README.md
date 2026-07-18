# egoemg_emgformer_middle_with_aug

EMGFormer Middle (256d, 8 heads, 6 layers) on EgoEMG, aggressive augmentation.

- **Eval date**: 2026-05-03
- **Checkpoint**: `checkpoints/egoemg_emgformer_middle_with_aug_mae0.2624_epoch045.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_middle_aggressive_egoemg`
- **With augmentation** (EgoEMG transform bug fixed)

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.233 | 0.234 ± 0.023 | 36 |
| user | 0.293 | 0.274 ± 0.038 | 6 |
| both | 0.311 | 0.306 ± 0.023 | 5 |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.287 | 0.262 ± 0.046 (6) | 0.284 ± 0.042 (52) |
| user/right | 0.299 | 0.286 ± 0.031 (6) | 0.298 ± 0.057 (52) |
| gesture/left | 0.223 | 0.224 ± 0.028 (36) | 0.180 ± 0.086 (17) |
| gesture/right | 0.244 | 0.244 ± 0.028 (36) | 0.206 ± 0.078 (17) |
| both/left | 0.307 | 0.299 ± 0.038 (5) | 0.294 ± 0.059 (16) |
| both/right | 0.314 | 0.313 ± 0.015 (5) | 0.303 ± 0.071 (16) |

Simple mean test_mae: **0.279**
