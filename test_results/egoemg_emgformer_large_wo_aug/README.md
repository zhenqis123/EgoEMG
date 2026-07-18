# egoemg_emgformer_large_wo_aug

EMGFormer Large (384d, 12 heads, 6 layers) on EgoEMG, no augmentation, from scratch.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_emgformer_large_wo_aug_mae0.2479_epoch000.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_large_aggressive_egoemg_wo_aug`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2019 | 0.2039 ± 0.0259 | 36 |
| user | 0.2987 | 0.2732 ± 0.0525 | 6 |
| both | 0.3125 | 0.3085 ± 0.0191 | 5 |
| **overall** | **0.2481** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2930 | 0.2668 ± 0.0526 (6) | 0.2898 ± 0.0492 (52) |
| user/right | 0.3046 | 0.2796 ± 0.0531 (6) | 0.3003 ± 0.0586 (52) |
| gesture/left | 0.1963 | 0.1990 ± 0.0286 (36) | 0.1314 ± 0.1118 (17) |
| gesture/right | 0.2081 | 0.2087 ± 0.0351 (36) | 0.1405 ± 0.1123 (17) |
| both/left | 0.3150 | 0.3079 ± 0.0292 (5) | 0.3027 ± 0.0748 (16) |
| both/right | 0.3099 | 0.3091 ± 0.0129 (5) | 0.3002 ± 0.0746 (16) |

Simple mean test_mae: **0.2712**
