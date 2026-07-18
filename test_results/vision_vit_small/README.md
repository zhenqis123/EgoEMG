# vision_vit_small

ViT-Small vision-only baseline for EgoEMG hand pose estimation (single-frame, no EMG).

- Architecture: ViT-Small → MLP head → 22 joint angles
- Checkpoint: epoch 179, val_mae=0.1053 (best in version_0)
- Config: `experiment/fusion/vision_vit_small`
- Fusion mode: vision_only

## Results

| split | hand | test_mae | fingertip (mm) |
|-------|------|----------|----------------|
| user | left | 0.133 | 16.4 |
| user | right | 0.122 | 15.5 |
| gesture | left | 0.088 | 11.6 |
| gesture | right | 0.086 | 11.2 |
| both | left | 0.130 | 15.8 |
| both | right | 0.121 | 14.7 |

**Simple mean test_mae: 0.113**

## Per-split summary (mean of left+right, ± std across users)

| Split | MAE (rad) | n_users |
|-------|-----------|---------|
| Gesture | 0.087 ± 0.023 | 36 |
| User | 0.127 ± 0.042 | 6 |
| Both | 0.125 ± 0.009 | 5 |
