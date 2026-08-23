<p align="center">
  <img src="images/dataset_stats.svg" width="100%" alt="EgoEMG dataset statistics">
</p>

<h1 align="center">EgoEMG</h1>

<p align="center">
  <b>A multimodal egocentric dataset with bilateral surface EMG and webcam vision
  for hand pose estimation</b><br>
  EMG-to-pose · vision-to-pose · EMG+vision fusion
</p>

<p align="center">
  <a href="#-release-status"><img src="https://img.shields.io/badge/release-v0.1.0-brightgreen" alt="Release status"></a>
  <a href="#-license"><img src="https://img.shields.io/badge/Code-MIT-blue" alt="Code license"></a>
  <a href="https://github.com/zhenqis123/EgoEMG/actions"><img src="https://img.shields.io/github/actions/workflow/status/zhenqis123/EgoEMG/main.yml?branch=main" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-blue" alt="Python">
</p>

<p align="center">
  [ <a href="#-setup">Setup</a> ·
    <a href="#-training">Training</a> ·
    <a href="#-visualization">Visualization</a> ·
    <a href="#-results-and-evaluation">Results</a> ·
    <a href="#-license">License</a> ·
    <a href="#-citing-egoemg">Citation</a> ]
</p>

---

## ✨ Highlights

- **🧠 EMG-to-pose** — EMGFormer (S/M/L): **13.9° Avg MAE** with the released M checkpoint.
- **👁️ Vision-to-pose** — ResNet / ViT on the egocentric webcam stream: **5.85°** with ResNet-18.
- **🔀 Fusion** — EMG + vision on identical center frames: **5.36°** with ResNet-50 fusion.
- **🎥 One-command visualization** — overlay videos with meshes and markers from the released memmap.

## ⚙️ Setup

```shell
conda env create -f environment.yml && conda activate emg2pose
pip install -e '.[viz]'
```

Download data and checkpoints with [Asset Setup](docs/ASSET_SETUP.md), then set
`export EMG2POSE_ROOT=/path/to/egoemg_assets`.

<details>
<summary><b>Dataset IMU note</b></summary>

Legacy copies downloaded before 2026-08-20 store EgoEMG IMU rows with the
acc/gyro halves swapped. Run
`scripts/prepare/fix_egoemg_imu_channel_order.py --memmap-dir <dir> --apply`
to repair, or re-download `imu.dat` / `manifest.json`. Details:
[Asset Setup §7](docs/ASSET_SETUP.md#7-imu-channel-order-fix-2026-08-20).
</details>

## 🚀 Training

```shell
# EMG-to-pose (EMGFormer-M)
python -m egoemg.train \
  experiment=emgformer/regression_egoemg \
  'trainer.devices=[0,1,2,3,4,5]' '+trainer.strategy=ddp' \
  batch_size=500

# Vision-only (ResNet-18)
python -m egoemg.train \
  experiment=fusion/vision_resnet18 \
  train=true eval=true 'trainer.devices=[0]'

# EMG+vision fusion
python -m egoemg.train \
  experiment=fusion/fusion_rn18_s_center_16ch_wl7790 \
  train=true eval=true 'trainer.devices=[0,1,2,3,4]'

# EMGFormer multi-task pretraining
python -m egoemg.train_pretrain \
  experiment=emgformer/pretrain_multitask \
  train=true eval=false 'trainer.devices=[0]'
```

> **Hardware:** the EMG example uses six GPUs, fusion five. For a single-GPU
> smoke run, set `trainer.devices=[0]` and reduce `batch_size`.

Experiment configs live in `config/experiment/{emgformer,fusion,emg2pose}/`.
For evaluation use `egoemg.test_analysis` — fusion configs default to
`train=true`, so the training entrypoint would start a new run.

## 🎥 Visualization

| Mode | Output |
|------|--------|
| `vision` | overlay MP4 + per-hand crop MP4s |
| `timeline` | EMG / joint-angle timeline PNG |
| `mesh` | world-space MANO/FK meshes (GLB) + occlusion renders |
| `fk_vs_mano` | FK vs MANO comparison |

```shell
python scripts/viz/visualize_dataset.py vision \
  --memmap-dir ${EMG2POSE_ROOT}/data/EgoEMG_unified_memmap \
  --allintra-root ${EMG2POSE_ROOT}/data/EgoEMG_allintra \
  --crops-dir ${EMG2POSE_ROOT}/data/EgoEMG_v2_crops \
  --data-root ${EMG2POSE_ROOT}/data \
  --output-dir /tmp/egoemg_vision_viz \
  --episode-id episode_000000 --stride 10 --max-frames 300
```

<p align="center">
  <img src="images/viz_example.jpg" width="70%" alt="vision-mode overlay: hand meshes, projected mocap markers, per-hand boxes">
</p>

## 📊 Results and evaluation

Bold = measured from the released checkpoints (single RTX 4090, commands
below); all other rows are paper-reported. MAE in degrees.

**EMG-to-pose on EgoEMG** (per-user mean ± std; Avg = per-sample-weighted):

| Method | Params | Gesture | User | Both | Avg. |
|--------|--------|---------|------|------|------|
| EMGFormer-S | 3.5M | 12.4 ± 1.5 | 16.0 ± 0.6 | 16.4 ± 1.5 | **14.2** |
| EMGFormer-M | 6.6M | 11.9 ± 1.6 | 16.0 ± 0.7 | 16.3 ± 1.5 | **13.9** |
| EMGFormer-L | 16.3M | 12.0 ± 1.6 | 16.1 ± 0.9 | 16.4 ± 1.3 | **14.0** |
| vEMG2Pose | 6.0M | 15.0 ± 1.4 | 16.3 ± 1.7 | 17.3 ± 1.3 | 15.9 |
| NeuroPose | 6.4M | 15.8 ± 1.2 | 15.7 ± 1.3 | 16.3 ± 0.7 | 16.1 |

**Vision and fusion on EgoEMG** (identical center frames; Frz./FT =
frozen/fine-tuned; Δavg = fusion gain):

| Backbone | Update | Vision Avg | Fusion Avg | Δavg |
|----------|--------|-----------|-----------|------|
| ResNet-18 | FT | **5.85** | 5.40 | +0.44 |
| ViT-S/14 | FT | **6.04** | **5.56** | +0.48 |
| ResNet-50 | Frz. | 5.27 | 5.19 | +0.09 |
| ResNet-152 | Frz. | 5.11 | 5.06 | +0.05 |
| ViT-B/14 | Frz. | 5.78 | 5.75 | +0.03 |
| ViT-L/14 | Frz. | 5.39 | 5.36 | +0.03 |
| WiLoR | Frz. | 4.73 | 4.68 | +0.04 |

*The bundle's ResNet-50 fusion checkpoint is fine-tuned with no paper row;
it measures 5.36°. The fine-tuned ResNet-18 fusion row is paper-reported only.*

### Evaluation

```shell
export EMG2POSE_ROOT=/absolute/path/to/dataset_root

# EgoEMG EMGFormer (per-group stats + overall)
python -m egoemg.test_analysis \
  experiment=emgformer/egoemg_emgformer_small \
  'checkpoint=checkpoints/egoemg_emgformer_small.ckpt'

# EMG2Pose benchmark
python -m egoemg.test_analysis \
  experiment=emgformer/emg2pose_emgformer_small \
  'checkpoint=checkpoints/emg2pose_emgformer_small.ckpt' \
  data_location=${EMG2POSE_ROOT}/data/emg_corpus/emg2pose_v3_memmap

# Vision-only ResNet-18
python -m egoemg.test_analysis \
  experiment=fusion/vision_resnet18 \
  'checkpoint=checkpoints/vision_resnet18.ckpt'

# Vision-only ViT-S/14
python -m egoemg.test_analysis \
  experiment=fusion/vision_vit_small \
  'checkpoint=checkpoints/vision_vit_small.ckpt'

# Fusion (ResNet-50 + 8ch EMG, WL 12000)
python -m egoemg.test_analysis \
  experiment=fusion/fusion_rn50_m_center_eval_released \
  'checkpoint=checkpoints/fusion_resnet_emgfusion_center.ckpt'

# Fusion (ViT-S/14 + 16ch EMG, WL 7790)
python -m egoemg.test_analysis \
  experiment=fusion/fusion_vits_s_center_eval_released \
  'checkpoint=checkpoints/fusion_vit_emgfusion_center.ckpt'
```

<details>
<summary><b>Evaluation notes</b></summary>

- Checkpoint filenames containing `=` cannot be passed through Hydra CLI;
  symlink to a `=`-free path.
- Vision/fusion evals use 2096/2082 center frames per hand. Longer windows
  drop edge frames (expected).
- `results.csv` lands in the Hydra run directory (`logs/<date>/...`).
- MAE is in **radians** in results.csv (`0.0944 rad ≈ 5.41°`); tables above
  are in degrees.
</details>

## 🗂️ Repository Layout

```text
egoemg/        models, datasets, training/eval, vendored UmeTrack FK
config/        Hydra experiments ({emgformer,fusion,emg2pose}/ + lineage/)
scripts/       data conversion, downloads, visualization, paper figures
experiments/   shell launchers for the paper's experiments
docs/          config architecture, asset setup, dataset notes
assets/        EMG layout figures and normalization statistics
```

## 📜 License

Code: **MIT**. Third-party material (incl. UmeTrack, CC-BY-NC-4.0) under
their own terms — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 📖 Citing EgoEMG

The paper is under review. Until a public version is available, cite this
repository and the exact commit — GitHub renders
[CITATION.cff](CITATION.cff).
