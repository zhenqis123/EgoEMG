# egoemg_emgformer_middle_center

EMGFormer Middle (256d, 8 heads, 6 layers) on EgoEMG, center-frame supervised, from pretrained checkpoint.

- **Eval date**: 2026-05-06
- **Checkpoint**: `checkpoints/egoemg_emgformer_middle_center_mae0.2595_epoch007.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_middle_aggressive_egoemg` + `center_supervised=true` + `center_target_only=true`
- **Pretrained from**: full-window middle checkpoint

## Per-hand per-group stats (6 splits)

| split | hand | samples | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|---|---|
| user | left | 830 | 0.2972 | 0.2726 ± 0.0470 (6) | 0.2929 ± 0.0456 (52) |
| user | right | 830 | 0.2944 | 0.2753 ± 0.0479 (6) | 0.2901 ± 0.0607 (52) |
| gesture | left | 1162 | 0.2188 | 0.2188 ± 0.0281 (36) | 0.1668 ± 0.0978 (17) |
| gesture | right | 1162 | 0.2314 | 0.2302 ± 0.0300 (36) | 0.1794 ± 0.0956 (17) |
| both | left | 161 | 0.3277 | 0.3180 ± 0.0344 (5) | 0.3152 ± 0.0708 (16) |
| both | right | 161 | 0.3052 | 0.3016 ± 0.0177 (5) | 0.2933 ± 0.0726 (16) |

Sample-weighted mean test_mae: **0.2592**
