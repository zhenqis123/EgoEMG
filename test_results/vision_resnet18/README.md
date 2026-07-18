# vision_resnet18

ResNet-18 vision-only baseline for EgoEMG hand pose estimation (single-frame, no EMG).

- Architecture: ResNet-18 backbone → MLP head → 22 joint angles
- Checkpoint: epoch 011, val_mae=0.1025 (best in version_9)
- Config: `experiment/fusion/vision_resnet_baseline`
- Fusion mode: vision_only

## Results

| split | hand | test_mae | fingertip (mm) |
|-------|------|----------|----------------|
| user | left | 0.122 | 15.3 |
| user | right | 0.113 | 14.6 |
| gesture | left | 0.090 | 11.7 |
| gesture | right | 0.087 | 11.3 |
| both | left | 0.123 | 14.9 |
| both | right | 0.114 | 13.7 |

**Simple mean test_mae: 0.108**

## Per-split summary (mean of left+right, ± std across users)

| Split | MAE (rad) | n_users |
|-------|-----------|---------|
| Gesture | 0.089 ± 0.023 | 36 |
| User | 0.118 ± 0.030 | 6 |
| Both | 0.118 ± 0.015 | 5 |
