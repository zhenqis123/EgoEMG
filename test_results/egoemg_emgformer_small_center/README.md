# egoemg_emgformer_small_center

EMGFormer Small (256d, 4 heads, 3 layers) on EgoEMG, center-frame supervised, from pretrained checkpoint.

- **Eval date**: 2026-05-06
- **Checkpoint**: `checkpoints/egoemg_emgformer_small_center_mae0.2532_epoch012.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_small_aggressive_egoemg` + `center_supervised=true` + `center_target_only=true`
- **Pretrained from**: `egoemg-small-epoch=259-val_mae=0.2551.ckpt` (full-window)

## Per-hand per-group stats (6 splits)

| split | hand | samples | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|---|---|
| user | left | 830 | 0.2915 | 0.2650 ± 0.0513 (6) | 0.2866 ± 0.0562 (52) |
| user | right | 830 | 0.2945 | 0.2774 ± 0.0379 (6) | 0.2921 ± 0.0591 (52) |
| gesture | left | 1162 | 0.2095 | 0.2105 ± 0.0279 (36) | 0.1592 ± 0.0951 (17) |
| gesture | right | 1162 | 0.2253 | 0.2245 ± 0.0306 (36) | 0.1700 ± 0.0981 (17) |
| both | left | 161 | 0.3091 | 0.3011 ± 0.0317 (5) | 0.3006 ± 0.0646 (16) |
| both | right | 161 | 0.2974 | 0.2937 ± 0.0151 (5) | 0.2857 ± 0.0728 (16) |

Sample-weighted mean test_mae: **0.2530**
