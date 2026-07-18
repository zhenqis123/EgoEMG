# egoemg_emgformer_small_wo_aug

EMGFormer Small (256d, 4 heads, 3 layers) on EgoEMG, no augmentation, from scratch.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_emgformer_small_wo_aug_mae0.2562_epoch190.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_small_aggressive_egoemg_wo_aug`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2219 | 0.2233 ± 0.0253 | 36 |
| user | 0.2932 | 0.2729 ± 0.0429 | 6 |
| both | 0.3076 | 0.3044 ± 0.0208 | 5 |
| **overall** | **0.2562** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2795 | 0.2569 ± 0.0477 (6) | 0.2777 ± 0.0430 (52) |
| user/right | 0.3071 | 0.2889 ± 0.0391 (6) | 0.3042 ± 0.0550 (52) |
| gesture/left | 0.2141 | 0.2162 ± 0.0275 (36) | 0.1727 ± 0.0818 (17) |
| gesture/right | 0.2298 | 0.2304 ± 0.0351 (36) | 0.1835 ± 0.0856 (17) |
| both/left | 0.2912 | 0.2872 ± 0.0162 (5) | 0.2828 ± 0.0606 (16) |
| both/right | 0.3239 | 0.3214 ± 0.0279 (5) | 0.3091 ± 0.0651 (16) |

Simple mean test_mae: **0.2743**
