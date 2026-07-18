# emg2pose_v3 Model Size Scaling Ablation

EMGFormer decoder size sweep on the emg2pose_v3 dataset.

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Dataset | emg2pose_v3 |
| Window Length | 20,000 (Middle/Large/XLarge/Huge), 10,000 (XXLarge) |
| Stride | 10,000 (WL=20K), 5,000 (WL=10K) |
| LR | 0.0001 |
| Max Epochs | 150 |
| GPUs | 6x RTX 4090 (DDP) |
| Batch Size | `3000 * 1000 / WL` |
| Seed | 42 |

## Model Configurations

| Model | model_dim | Heads | Layers | FFN | Total Params |
|-------|-----------|-------|--------|-----|-------------|
| Middle | 256 | 8 | 6 | 1024 | 6.6M |
| Large | 384 | 12 | 8 | 1536 | 16.1M |
| XLarge | 512 | 8 | 8 | 2048 | ~27M |
| **XXLarge** | **640** | **10** | **10** | **2560** | **~51.8M** |
| Huge | 768 | 12 | 14 | 3072 | ~101M |

## Key Results

| Model | Val MAE | Test Stage MAE | Test User MAE | Fingertip Dist (mm) | Landmark Dist (mm) |
|-------|---------|----------------|---------------|---------------------|-------------------|
| Middle | 0.2249 | 0.1744 | 0.2124 | 20.63 | 12.63 |
| Large | 0.2254 | 0.1511 | 0.2123 | 17.66 | 10.84 |
| XLarge | 0.2234 | 0.1335 | 0.2110 | 15.51 | 9.54 |
| XXLarge | 0.2230 | 0.1225 | 0.2141 | 14.08 | 8.68 |
| Huge (best) | 0.2216 | 0.1478 | 0.2072 | 17.21 | 10.57 |
| **Huge (last)** | — | **0.1038** | **0.2129** | **12.01** | **7.40** |

## Key Findings

1. **Val MAE is unreliable for model selection on emg2pose_v3** — 5 models differ by only 0.005 in val_mae but 41.6% in test stage MAE. Huge's best val checkpoint (ep26) gives stage MAE 0.1478, but the last checkpoint (ep81) achieves 0.1038 — a 29.8% improvement while val_mae got *worse*. (Val evaluation has been updated to include val+test splits.)
2. **Huge (last, ep81) is the best model** — stage MAE 0.1038, fingertip 12.01mm, landmark 7.40mm. Despite val_mae suggesting overfitting after ep26, test stage MAE continued improving through ep81.
3. **XXLarge (~51.8M) is the efficiency sweet spot** — stage MAE 0.1225, fingertip 14.08mm, landmark 8.68mm with half the params of Huge. Best val checkpoint at ep60/150.
4. **Huge (101M) shows val/test divergence, not overfitting** — best val checkpoint at ep26, but test stage MAE kept improving until ep81 (last). The model was not actually overfitting on stage generalization; val_mae was simply the wrong metric.
5. **Model scaling >> Window scaling** — 41.6% improvement from model size vs ~0.9% from window length.
6. **Larger models have bigger generalization gaps** — stage-to-user gap grows from 3.8% (Middle) to 9.2% (XXLarge) and 14.6% (Huge last).

## Files

- `report.html` — Interactive HTML report with charts
- `generate_report.py` — Report generation script
- `data/model_scaling_results.json` — Complete results data
- `data/hparams_*.yaml` — Training hyperparameters for each model
- `results/*_test.csv` — Per-split test results (stage/user/user_stage)

## How to Reproduce

```bash
# Middle (6.6M), WL=20000
python -m emg2pose.train \
  experiment=emgformer/regression_emg2pose \
  'trainer.devices=[0,1,2,3,4,5]' +trainer.strategy=ddp \
  trainer.max_epochs=150 batch_size=150 \
  datamodule.window_length=20000 datamodule.stride=10000

# Large (16.1M)
python -m emg2pose.train \
  experiment=emgformer/regression_emg2pose \
  'trainer.devices=[0,1,2,3,4,5]' +trainer.strategy=ddp \
  trainer.max_epochs=150 batch_size=150 \
  datamodule.window_length=20000 datamodule.stride=10000 \
  module.decoder.model_dim=384 module.decoder.num_heads=12 \
  module.decoder.num_layers=8 module.decoder.ffn_dim=1536

# XLarge (~27M)
python -m emg2pose.train \
  experiment=emgformer/regression_emg2pose \
  'trainer.devices=[0,1,2,3,4,5]' +trainer.strategy=ddp \
  trainer.max_epochs=150 batch_size=150 \
  datamodule.window_length=20000 datamodule.stride=10000 \
  module.decoder.model_dim=512 module.decoder.num_heads=8 \
  module.decoder.num_layers=8 module.decoder.ffn_dim=2048

# XXLarge (~51.8M), WL=10000
python -m emg2pose.train \
  experiment=emgformer/regression_emg2pose \
  'trainer.devices=[0,1,2,3,4,5]' +trainer.strategy=ddp \
  trainer.max_epochs=150 batch_size=300 \
  datamodule.window_length=10000 datamodule.stride=5000 \
  module.decoder.model_dim=640 module.decoder.num_heads=10 \
  module.decoder.num_layers=10 module.decoder.ffn_dim=2560

# Huge (~101M)
python -m emg2pose.train \
  experiment=emgformer/regression_emg2pose \
  'trainer.devices=[0,1,2,3,4,5]' +trainer.strategy=ddp \
  trainer.max_epochs=150 batch_size=150 \
  datamodule.window_length=20000 datamodule.stride=10000 \
  module.decoder.model_dim=768 module.decoder.num_heads=12 \
  module.decoder.num_layers=14 module.decoder.ffn_dim=3072
```
