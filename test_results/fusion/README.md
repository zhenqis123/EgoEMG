# Fusion Experiments

EMG + Vision fusion results on EgoEMG.

## F-RN18+S (version_12, center_supervised, all trainable)

Model: ResNet18 + EMGFormer Small
Checkpoint: `logs/fusion/resnet_small_emgfusion_center/version_12/checkpoints/resnet-small-centerfusion-epoch=002-val_mae=0.0978.ckpt`
Params: 15.5M

| Split | Hand | MAE (rad) | MAE (°) | Fingertip (mm) | Landmark (mm) |
|-------|------|-----------|---------|-----------------|---------------|
| user | left | 0.1202 | 6.89 | 15.10 | 9.45 |
| user | right | 0.1110 | 6.36 | 14.30 | 8.96 |
| gesture | left | 0.0837 | 4.80 | 10.90 | 6.81 |
| gesture | right | 0.0813 | 4.66 | 10.73 | 6.66 |
| both | left | 0.1212 | 6.95 | 14.69 | 9.29 |
| both | right | 0.1107 | 6.35 | 13.65 | 8.70 |
| **Mean** | | **0.0978** | **5.60** | **12.56** | **8.31** |

Per-split (sample-weighted, °): gesture=4.7±1.3, user=6.6±1.4, both=6.6±0.8

## F-ViTS+S (version_7, center_supervised, all trainable)

Model: ViT-Small (DINOv2) + EMGFormer Small
Checkpoint: `logs/fusion/vit_small_emgfusion_center/version_7/checkpoints/vit-small-centerfusion-epoch=088-val_mae=0.0968.ckpt`
Params: 25.9M

| Split | Hand | MAE (rad) | MAE (°) | Fingertip (mm) | Landmark (mm) |
|-------|------|-----------|---------|-----------------|---------------|
| user | left | 0.1232 | 7.06 | 14.97 | 9.42 |
| user | right | 0.1147 | 6.57 | 14.25 | 8.99 |
| gesture | left | 0.0786 | 4.50 | 10.45 | 6.53 |
| gesture | right | 0.0769 | 4.41 | 10.30 | 6.41 |
| both | left | 0.1236 | 7.08 | 14.86 | 9.40 |
| both | right | 0.1120 | 6.42 | 13.21 | 8.46 |
| **Mean** | | **0.0966** | **5.54** | **12.28** | **8.20** |

Per-split (sample-weighted, °): gesture=4.5±1.3, user=6.8±1.2, both=6.8±0.6

## Old Results (version_9, superseded)

Model: ResNet18 + EMGFormer Small
Checkpoint: `test_results/fusion/checkpoints/best.ckpt` (epoch=198, val_mae=0.0984)
CSV: `version_9_test.csv` (no per-user std)

## V-WiLoR (WiLoR ViT-Huge, vision-only, center-frame)

Model: WiLoR ViT-Huge (631.6M), MLP head (1280→512→22)
Checkpoint: `logs_temp/fusion/vision_wilor_vit/version_2/checkpoints/wilor-vit-epoch=011-val_mae=0.0840.ckpt`
CSV: `wilor_vit_ep11.csv`
Config: `config/experiment/fusion/vision_wilor_vit.yaml`

| Split | Hand | MAE (rad) | Per-user mean | Per-user std | Users |
|-------|------|-----------|---------------|--------------|-------|
| gesture | left | 0.0730 | 0.0735 | 0.0219 | 36 |
| gesture | right | 0.0700 | 0.0713 | 0.0220 | 36 |
| user | left | 0.1000 | 0.1192 | 0.0452 | 6 |
| user | right | 0.0970 | 0.1137 | 0.0423 | 6 |
| both | left | 0.1010 | 0.0992 | 0.0137 | 5 |
| both | right | 0.0964 | 0.0942 | 0.0095 | 5 |

Sample-weighted overall MAE: **0.0839 rad (4.81°)**.
Per-split (°): gesture=4.1±1.3, user=5.6±2.5, both=5.7±0.7.

## V-WiLoR Scratch (ViT-Huge, vision-only, from-scratch training)

Model: WiLoR ViT-Huge (631.6M), MLP head (1280→512→22), **no pretrained init**
Checkpoint: `logs_temp/fusion/vision_wilor_vit_scratch/version_0/checkpoints/wilor-vit-scratch-epoch=044-val_mae=0.1281.ckpt`
CSV: `wilor_vit_scratch_ep44.csv`
Config: `config/experiment/fusion/vision_wilor_vit_scratch.yaml`
Training: AdamW lr=5e-5, cosine annealing, 50 epochs, batch_size=256

| Split | Hand | MAE (rad) | MAE (°) | Fingertip (mm) | Landmark (mm) |
|-------|------|-----------|---------|-----------------|---------------|
| user | left | 0.1537 | 8.80 | 19.15 | 11.98 |
| user | right | 0.1422 | 8.15 | 18.93 | 11.78 |
| gesture | left | 0.1136 | 6.51 | 15.40 | 9.51 |
| gesture | right | 0.1087 | 6.23 | 14.48 | 9.03 |
| both | left | 0.1526 | 8.75 | 19.04 | 11.97 |
| both | right | 0.1414 | 8.10 | 18.06 | 11.33 |
| **Mean** | | **0.1280** | **7.33** | **17.34** | **10.60** |

Per-split (sample-weighted, °): gesture=6.4±1.3, user=8.5±1.8, both=8.4±0.5

### Scratch vs WiLoR-initialized comparison

| Split | V-WiLoR (ep22, init) | V-WiLoR (ep44, scratch) | Δ |
|-------|---------------------|------------------------|---|
| Gesture | 3.9° | 6.4° | +2.4° |
| User | 5.6° | 8.5° | +2.8° |
| Both | 5.7° | 8.4° | +2.7° |
| **Avg** | **4.8°** | **7.3°** | **+2.5°** |

WiLoR pretraining provides a consistent ~2.5° MAE improvement across all splits, confirming that the hand-specialized MANO initialization is important for convergence quality.

Init: WiLoR pretrained (ViTPose → WiLoR MANO checkpoint `wilor_final.ckpt`).
ViT-Huge: 1280-dim, 32 layers, 16 heads, patch=16, input=256×192 (cropped from 256×256).

## Delta Contribution

See `architecture.md` for full pipeline documentation and delta contribution analysis.
