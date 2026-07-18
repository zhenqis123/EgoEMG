# egoemg_emgformer_middle_wo_aug

EMGFormer Middle (256d, 8 heads, 6 layers) on EgoEMG, no augmentation, from scratch.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_emgformer_middle_wo_aug_mae0.2482_epoch108.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_middle_aggressive_egoemg_wo_aug`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2035 | 0.2060 ± 0.0279 | 36 |
| user | 0.2979 | 0.2850 ± 0.0254 | 6 |
| both | 0.3076 | 0.3034 ± 0.0158 | 5 |
| **overall** | **0.2482** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2926 | 0.2700 ± 0.0439 (6) | 0.2902 ± 0.0356 (52) |
| user/right | 0.3035 | 0.3000 ± 0.0096 (6) | 0.3011 ± 0.0535 (52) |
| gesture/left | 0.1980 | 0.2003 ± 0.0293 (36) | 0.1410 ± 0.1032 (17) |
| gesture/right | 0.2096 | 0.2116 ± 0.0358 (36) | 0.1487 ± 0.1032 (17) |
| both/left | 0.3031 | 0.2965 ± 0.0239 (5) | 0.2922 ± 0.0653 (16) |
| both/right | 0.3119 | 0.3103 ± 0.0086 (5) | 0.3044 ± 0.0723 (16) |

Simple mean test_mae: **0.2698**
