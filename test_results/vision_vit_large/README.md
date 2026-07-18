# vision_vit_large

ViT-Large vision-only baseline for EgoEMG hand pose estimation (single-frame, no EMG).

- Architecture: ViT-Large → MLP head → 22 joint angles
- Checkpoint: epoch 127, val_mae=0.0940 (best in version_0)
- Config: `experiment/fusion/vision_vit_large`
- Fusion mode: vision_only

## Results

| split | hand | test_mae | fingertip (mm) |
|-------|------|----------|----------------|
| user | left | 0.121 | 14.4 |
| user | right | 0.111 | 13.6 |
| gesture | left | 0.076 | 9.9 |
| gesture | right | 0.075 | 9.7 |
| both | left | 0.119 | 14.0 |
| both | right | 0.112 | 13.4 |

**Simple mean test_mae: 0.102**

## Per-split summary (mean of left+right, ± std across users)

| Split | MAE (rad) | n_users |
|-------|-----------|---------|
| Gesture | 0.076 ± 0.023 | 36 |
| User | 0.116 ± 0.038 | 6 |
| Both | 0.116 ± 0.013 | 5 |
