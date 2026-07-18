# vision_vit_base

ViT-Base vision-only baseline for EgoEMG hand pose estimation (single-frame, no EMG).

- Architecture: ViT-Base → MLP head → 22 joint angles
- Checkpoint: epoch 123, val_mae=0.1010 (best in version_0)
- Config: `experiment/fusion/vision_vit_base`
- Fusion mode: vision_only

## Results

| split | hand | test_mae | fingertip (mm) |
|-------|------|----------|----------------|
| user | left | 0.127 | 15.3 |
| user | right | 0.116 | 14.8 |
| gesture | left | 0.085 | 11.0 |
| gesture | right | 0.083 | 10.8 |
| both | left | 0.126 | 15.2 |
| both | right | 0.115 | 14.1 |

**Simple mean test_mae: 0.109**

## Per-split summary (mean of left+right, ± std across users)

| Split | MAE (rad) | n_users |
|-------|-----------|---------|
| Gesture | 0.084 ± 0.023 | 36 |
| User | 0.122 ± 0.040 | 6 |
| Both | 0.120 ± 0.011 | 5 |
