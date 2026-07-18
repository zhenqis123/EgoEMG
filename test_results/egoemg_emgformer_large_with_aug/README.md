# egoemg_emgformer_large_with_aug

EMGFormer Large (384d, 12 heads, 8 layers) on EgoEMG, aggressive augmentation.

- **Eval date**: 2026-05-03
- **Checkpoint**: `checkpoints/egoemg_emgformer_large_with_aug_mae0.2619_epoch061.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_large_aggressive_egoemg`
- **With augmentation** (EgoEMG transform bug fixed)

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.231 | 0.233 ± 0.024 | 36 |
| user | 0.294 | 0.272 ± 0.045 | 6 |
| both | 0.313 | 0.307 ± 0.021 | 5 |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.291 | 0.267 ± 0.048 (6) | 0.287 ± 0.050 (52) |
| user/right | 0.297 | 0.278 ± 0.044 (6) | 0.294 ± 0.063 (52) |
| gesture/left | 0.221 | 0.223 ± 0.027 (36) | 0.170 ± 0.094 (17) |
| gesture/right | 0.242 | 0.243 ± 0.031 (36) | 0.194 ± 0.089 (17) |
| both/left | 0.311 | 0.302 ± 0.034 (5) | 0.300 ± 0.071 (16) |
| both/right | 0.315 | 0.312 ± 0.017 (5) | 0.301 ± 0.074 (16) |

Simple mean test_mae: **0.279**
