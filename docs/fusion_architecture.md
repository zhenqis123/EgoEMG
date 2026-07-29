# EMG + Vision Fusion Architecture

## Overview

Post-decoder EMG+Vision fusion model (`MidFusionPoseFormer`) takes an EMG window and a single
vision frame, extracts features from each modality independently, processes EMG through a
TransformerDecoder, then fuses decoder output with vision features before the prediction head.

**Training entrypoint:** `python -m emg2pose.train`

**Config:** `experiment=fusion/vision_resnet_small_emgfusion_center` (see `config/experiment/fusion/`)

**Source:** `emg2pose/models/modules/mid_fusion.py`

---

## Architecture Diagram

```
EMG: (B, 16, 7790)                          Vision: (B, 3, 256, 256)
      │                                            │
      ▼                                            ▼
 ┌───────────────┐                          ┌───────────────────┐
 │ TDS Featurizer│                          │  WiLoR ViT (frozen)│
 │  tds_slim     │                          │  ViT-L, 32 blocks │
 │  (1.7M params)│                          │  (630M, frozen)   │
 └───────┬───────┘                          └────────┬──────────┘
         │ (B, 256, 146)                            │ (B, 1280)
         │                                    ┌─────┴─────┐
         │                                    │ vision_proj│
         │                                    │ Linear     │
         ▼                                    │ 1280→256   │
 ┌──────────────┐                        ┌────┴─────┐
 │ Transformer  │                        │ expand(T)│
 │ Decoder (6L) │                        │ (B,256,146)│
 │ 8 heads      │                        └────┬─────┘
 │ (4.8M)       │                             │
 └──────┬───────┘                             │
        │ (B, 256, 146)                       │
        │                  ┌──────────────────┘
        │                  ▼
        │            ┌──────────┐
        └──────────► │  concat  │ (B, 512, 146)
                     └────┬─────┘
                          ▼
                     ┌──────────┐
                     │fusion_proj│
                     │ Conv1d k=1│
                     │512→256   │
                     └────┬─────┘
                          │ (B, 256, 146)
                          ▼
                     ┌──────────┐
                     │ MLPHead  │
                     │ 256→22×T │
                     │ (142K)   │
                     └────┬─────┘
                          │ (B, 22, 146)
                          ▼
                   Joint Angles (22 joints × T timesteps)
```

---

## Component Details

### 1. EMG Branch — TDS Featurizer (`tds_slim`)

**Config:** `config/module/featurizer/tds_slim.yaml`

Same architecture as `regression_emgformer_middle_aggressive`.

| Stage | Module | Input | Output | Kernel | Stride |
|-------|--------|-------|--------|--------|--------|
| Conv1 | Conv1dBlock | (B, 16, 7790) | (B, 256, 1558) | k=11 | s=5 |
| Conv2 | Conv1dBlock | (B, 256, 1558) | (B, 256, 312) | k=5 | s=2 |
| TDS1 | TdsStage (1 block) | (B, 256, 312) | (B, 256, 75) | k=9 | s=5 |
| TDS2 | TdsStage (1 block) | (B, 256, 75) | (B, 256, 146) | k=3 | s=1 |

- Each TDS block: 2D conv (temporal + channel) + SE attention + residual
- SE: reduction=4, mode=global
- **Left context:** 34 samples (receptive field of conv + TDS stages)
- **Output:** `(B, 256, 146)` — 146 timesteps, 256 channels
- **Params:** ~1.7M (fully trainable)

### 2. Vision Branch — WiLoR ViT Backbone

**Config:** WiLoR model config from `WiLoR/pretrained_models/model_config.yaml`

| Stage | Module | Input | Output | Params |
|-------|--------|-------|--------|--------|
| Patch Embed | PatchEmbed | (B, 3, 256, 256) | (B, 1280, 257) | frozen |
| Positional Embedding | nn.Parameter | — | (B, 257, 1280) | frozen |
| Transformer Blocks | 32 × Block | (B, 257, 1280) | (B, 257, 1280) | frozen |
| Last Norm | LayerNorm | (B, 257, 1280) | (B, 257, 1280) | frozen |
| MANO heads | Linear layers | (B, 1280) | MANO params | frozen |

- Vision backbone loaded from WiLoR checkpoint (`wilor_final.ckpt`)
- 32 blocks all frozen (`frozen_block_end: 32`)
- MANO token embeddings + decode heads also frozen
- **Feature extraction:** global average pool over spatial dimensions → `(B, 1280)`

**Freeze strategy (`simple`):**
- Frozen: `blocks[0:32]`, `patch_embed`, `pos_embed`, `last_norm`, MANO tokens
- Total frozen: ~630M params
- All ViT parameters have `requires_grad=False`

### 3. Vision Projection

| Layer | Input | Output |
|-------|-------|--------|
| `vision_proj`: Linear(1280, 256) | (B, 1280) | (B, 256) |
| `expand` (repeat along T) | (B, 256) | (B, 256, 146) |

- Vision feature is a single-frame global descriptor, broadcast to match EMG temporal length
- Invalid vision frames (no crop available) are zeroed out via `vision_valid_mask`
- **Params:** ~327K (trainable)

### 4. Transformer Decoder (`middle`)

**Config:** `config/module/decoder/transformer/preset/middle.yaml`

| Param | Value |
|-------|-------|
| `model_dim` | 256 |
| `num_heads` | 8 |
| `num_layers` | 6 |
| `ffn_dim` | 1024 |
| `causal` | false (non-autoregressive) |
| `pos_encoding` | RoPE |
| `dropout` | 0.15 |
| `norm_first` | true (pre-norm) |

- Non-causal: decoder attends to full temporal context
- RoPE: rotary positional embeddings for relative position awareness
- **EMG-only:** vision features are NOT fed into the decoder
- **Params:** ~4.8M (fully trainable)

### 5. Fusion (post-decoder concat)

| Layer | Input | Output |
|-------|-------|--------|
| Concat | Decoder(B,256,146) + Vision(B,256,146) | (B, 512, 146) |
| `fusion_proj`: Conv1d(512, 256, k=1) | (B, 512, 146) | (B, 256, 146) |

- Channel-wise concatenation of decoder output and vision features
- 1x1 convolution for dimensionality reduction
- **Params:** ~131K (trainable)

### 6. Prediction Head (MLPHead)

| Layer | Output |
|-------|--------|
| Linear(256, 512) | (B, 512, 146) |
| ReLU | (B, 512, 146) |
| Dropout(0.1) | (B, 512, 146) |
| Linear(512, 22) | (B, 22, 146) |

- 22 output channels: 20 joint angles + 2 wrist angles (pitch, yaw)
- **Params:** ~142K (fully trainable)

---

## Parameter Summary

| Component | Params | Trainable | Frozen |
|-----------|--------|-----------|--------|
| TDS Featurizer | 1.7M | 1.7M | 0 |
| Transformer Decoder | 4.8M | 4.8M | 0 |
| MLP Head | 142K | 142K | 0 |
| Vision Backbone (ViT) | 630M | 0 | 630M |
| Vision Projection | 327K | 327K | 0 |
| Fusion Projection | 131K | 131K | 0 |
| **Total** | **638M** | **7.1M** | **630M** |

---

## Input / Output

### Inputs

| Name | Shape | Description |
|------|-------|-------------|
| `emg` | (B, 16, 7790) | 16-channel EMG, 2000Hz, 3.895s window |
| `vision_img` | (B, 3, 256, 256) | Single pre-cropped hand image at window center |
| `vision_valid_mask` | (B,) or (B, 1) | Whether vision crop is available |

### Outputs

| Name | Shape | Description |
|------|-------|-------------|
| `preds` | (B, 22, 146) | Predicted joint angles (radians) |
| `targets` | (B, 22, 146) | GT joint angles interpolated to match T' |
| `mask` | (B, 146) | Valid frame mask (bool) |

### Target Handling

- GT joint angles are `(B, 22, 7280)` at 2000Hz. After slicing off left context (34 samples) and
  linear interpolation to match the model's output temporal dimension (146), targets become `(B, 22, 146)`.
- Valid mask is aligned via nearest-neighbor interpolation.

---

## Dataset

**Dataset:** `EgoEmgMemmapDataset` with pre-crop LMDBs

**Pre-crops:** `data/EgoEMG_v2_crops/` — per-episode LMDB files containing
JPEG-encoded 256×256 hand crops at `FF:NNNNNNNN_H` keys (frame index + hand code).

**EMG data:** `data/EgoEMG_v2_memmap/` — EgoEMG v2 memmap with 8-channel EMG layout
(`emg2pose_interpolate16`), normalized using per-dataset statistics.

**Vision fields:** `per_episode_crops_dir` enabled → bypasses decord video decoding and
calibration-based bbox computation. Reads pre-cropped JPEGs directly from LMDB, applies
ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

---

## Loss

**Objective:** MAE (L1) loss on joint angles.

```
L = L1Loss(pred[mask], target[mask])
```

Additional metrics logged: per-finger MAE, velocity, accuracy, fingertip distance, landmark distance.

**Optimizer:** Adam, lr=5e-4, weight_decay=1e-4

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Batch size | 64 |
| Max epochs | 100 |
| Precision | bf16-mixed |
| Devices | GPU(s) as specified |
| Gradient clip | 1.0 |
| Checkpoint | Top-3 by val_mae + last |
| Early stopping | patience=30 on val_mae |

---

## Quick Start

```bash
# Full training
python -m emg2pose.train \
  experiment=fusion/vision_resnet_small_emgfusion_center \
  trainer.devices=[1] \
  num_workers=8

# Quick dry-run (1 epoch, no workers)
python -m emg2pose.train \
  experiment=fusion/vision_resnet_small_emgfusion_center \
  max_epochs=1 trainer.max_epochs=1 \
  trainer.devices=[1] \
  datamodule.stride=7790 \
  datamodule.val_test_stride=7790 \
  num_workers=0
```
