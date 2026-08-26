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
  <img src="https://img.shields.io/badge/release-v0.1.0-brightgreen" alt="Release">
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

- **Multi-modal Dataset And Benchmark** — Provides synchronized bilateral surface EMG and egocentric webcam video for hand pose estimation.
- **🧠 EMG-to-pose** — EMGFormer (S/M/L): **13.8°** Avg MAE on EgoEMG (M);
  **12.3°** User+Stage MAE on the EMG2Pose benchmark (S) — **ahead of all prior methods**,
  including Position/Velocity MT (14.6°), CLDM (14.7°), and emg2pose (15.6°).
- **👁️ Vision-to-pose** — ResNet / ViT on the egocentric webcam stream: **5.85°** with ResNet-18.
- **🔀 Fusion** — EMG + vision on identical center frames: **5.41°** with ResNet-18 fusion, a 7.5% improvement with only 3M more parameters compared to ResNet-18 vision-only baseline.

## 🎯 Headline results

MAE in degrees. \* = released-checkpoint values (reproduce with the commands
in [Results and evaluation](#-results-and-evaluation)); unstarred values are
paper-reported.

| Task | Ours | Best baseline |
|------|------|---------------|
| EMG-to-pose on EgoEMG | **13.8°** (EMGFormer-M) \* | 15.8° (emg2pose) |
| EMG-to-pose on EMG2Pose | **12.3°** (EMGFormer-S) | 14.6° (Position/Velocity MT) |
| Vision-to-pose on EgoEMG | **5.85°** (ResNet-18) \* | — |
| EMG+vision fusion on EgoEMG | **5.41°** (ResNet-18) \* | 5.85° (vision-only) |

Full per-method and per-split breakdowns are in the
[Results and evaluation](#-results-and-evaluation) section below.

## ⚙️ Setup

```shell
conda env create -f environment.yml && conda activate egoemg
pip install -e '.[viz]'
```

Download data and checkpoints with [Asset Setup](docs/ASSET_SETUP.md), then set
`export EGOEMG_ROOT=/path/to/egoemg_assets`.

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
For evaluation use `egoemg.test_analysis` (fusion configs default `train=true`).
The EMG+vision fusion model (`MidFusionPoseFormer`) is detailed in
[Fusion Architecture](docs/fusion_architecture.md).

## 🎥 Visualization

| Mode | Output |
|------|--------|
| `vision` | overlay MP4 + per-hand crop MP4s |
| `timeline` | EMG / joint-angle timeline PNG |
| `mesh` | world-space MANO/FK meshes (GLB) + occlusion renders |
| `fk_vs_mano` | FK vs MANO comparison |

```shell
python scripts/viz/visualize_dataset.py vision \
  --memmap-dir ${EGOEMG_ROOT}/data/EgoEMG_full_memmap \
  --allintra-root ${EGOEMG_ROOT}/data/EgoEMG_videos \
  --crops-dir ${EGOEMG_ROOT}/data/EgoEMG_crops \
  --data-root ${EGOEMG_ROOT}/data \
  --output-dir /tmp/egoemg_vision_viz \
  --episode-id episode_000000 --stride 10 --max-frames 300
```

<p align="center">
  <video src="images/episode_000020_vision.mp4" width="70%" controls muted playsinline>
    <img src="images/viz_example.jpg" width="70%" alt="vision-mode overlay: hand meshes, projected mocap markers, per-hand boxes">
  </video>
</p>

### Preview / small dataset

Every workflow above also runs end-to-end on a small **preview shard** instead
of the full `EgoEMG_full_memmap`. It is a 3-episode, **v3-schema** memmap in the
same **flat** layout as the published `dataset_egoemg_unified`, so it drops in as
a mini dataset root: point the same commands at its `data/memmap_data`.

Download it once (two mirrors — Google Drive is the default and writes straight
into the target dir; Baidu Netdisk drops the package under `./download/` and
needs a logged-in `baidupcs`):

```shell
# Google Drive (default)
bash scripts/download/download_egoemg_data.sh \
  "$EGOEMG_ROOT/data/dataset_egoemg_preview"

# Baidu Netdisk (/EgoEMG_release/dataset_egoemg_preview)
bash scripts/download/download_egoemg_data.sh --source baidupcs \
  "$EGOEMG_ROOT/data/dataset_egoemg_preview"
```

```shell
# EMG-to-pose eval (EMGFormer-M, 8-ch) on the shard
python -m egoemg.test_analysis \
  experiment=emgformer/egoemg_emgformer_middle \
  'checkpoint=checkpoints/egoemg_emgformer_middle.ckpt' \
  egoemg_unified_memmap_dir=$EGOEMG_ROOT/data/dataset_egoemg_preview/data/memmap_data \
  'trainer.devices=[0]' \
  datamodule.per_dataset_norm_stats_path=assets/per_dataset_norm_stats_unified.json

# Vision overlay (episode_000028 is in the shard; needs MANO model files per
# docs/ASSET_SETUP.md §3 — set WILOR_PATH or pass --mano-model-path)
python scripts/viz/visualize_dataset.py vision \
  --memmap-dir $EGOEMG_ROOT/data/dataset_egoemg_preview/data/memmap_data \
  --allintra-root $EGOEMG_ROOT/data/dataset_egoemg_preview/data/webcam_videos \
  --crops-dir $EGOEMG_ROOT/data/dataset_egoemg_preview/data/pre-crop_webcam_videoframes \
  --data-root $EGOEMG_ROOT/data/dataset_egoemg_preview/data \
  --episode-id episode_000028 --stride 10 --max-frames 300 \
  --mano-model-path $WILOR_PATH/mano_data/models

# Smoke training (1 epoch, 2 batches; small batch_size for a single GPU)
python -m egoemg.train experiment=emgformer/egoemg_emgformer_small \
  egoemg_unified_memmap_dir=$EGOEMG_ROOT/data/dataset_egoemg_preview/data/memmap_data \
  'trainer.devices=[0]' 'trainer.max_epochs=1' batch_size=8 \
  '+trainer.limit_train_batches=2' '+trainer.limit_val_batches=0' \
  datamodule.per_dataset_norm_stats_path=assets/per_dataset_norm_stats_unified.json
```

> **Why the replay may look static** — the `generated_mano_*_pose` labels are
> zero-filled on rows where `generated_label_valid=false` (~32% of rows) and
> also contain static plateaus, so replaying those MANO meshes does not track
> the reference video. Replay the **supervised joint angles + wrist
> (`generated_joint_angles_*`)** and filter to valid frames instead. Build or
> inspect the shard with the [Asset Setup](docs/ASSET_SETUP.md#preview-package)
> script.

## 📊 Results and evaluation

<details>
<summary>Full benchmark tables</summary>

\* marks values measured with the released checkpoints (commands below);
other cells are paper-reported. MAE in degrees.

**EMG-to-pose on EgoEMG** (per-user mean ± std; Avg = per-sample-weighted):

| Method | Params | Gesture | User | Both | Avg. |
|--------|--------|---------|------|------|------|
| EMGFormer-S | 3.5M | 12.3 ± 1.5 | 16.0 ± 0.6 | 16.3 ± 1.5 | 14.1 \* |
| EMGFormer-M | 6.6M | 11.7 ± 1.6 | 15.9 ± 0.6 | 16.4 ± 1.5 | 13.8 \* |
| EMGFormer-L | 16.3M | 11.9 ± 1.6 | 16.0 ± 0.8 | 16.4 ± 1.2 | 13.9 \* |
| emg2pose | 3.0M | 15.5 ± 1.3 | 14.8 ± 2.9 | 16.3 ± 0.6 | 15.8 |
| vEMG2Pose | 6.0M | 15.0 ± 1.4 | 16.3 ± 1.7 | 17.3 ± 1.3 | 15.9 |
| NeuroPose | 6.4M | 15.8 ± 1.2 | 15.7 ± 1.3 | 16.3 ± 0.7 | 16.1 |
| SensingDynamics | 1.0M | 16.2 ± 1.1 | 16.4 ± 0.3 | 16.7 ± 0.8 | 16.4 |

**EMG-to-pose on the EMG2Pose benchmark** (per-user mean ± std across the
User / Stage / User+Stage test splits):

| Method | Params | User | Stage | User+Stage |
|--------|--------|------|-------|------------|
| EMGFormer-S | 3.5M | 12.5 ± 1.1 | 11.1 ± 1.2 | 12.3 ± 1.1 |
| EMGFormer-M | 6.6M | 12.4 ± 1.1 | 10.2 ± 1.1 | 12.4 ± 1.1 |
| EMGFormer-L | 16.3M | 12.3 ± 1.1 | 9.3 ± 1.1 | 12.3 ± 1.1 |
| emg2pose | 3.0M | 12.6 ± 1.3 | 15.2 ± 1.6 | 15.6 ± 1.3 |
| vEMG2Pose | 6.0M | 12.2 ± 1.3 | 15.2 ± 1.6 | 15.8 ± 1.4 |
| NeuroPose | 6.4M | 13.2 ± 1.1 | 17.2 ± 1.7 | 17.5 ± 1.5 |
| SensingDynamics | 1.0M | 15.5 ± 1.4 | 18.8 ± 1.6 | 18.7 ± 1.6 |
| Position MT | 6.0M | 11.5 ± 1.2 | 14.0 ± 1.6 | 14.6 ± 1.3 |
| Velocity MT | 6.0M | 11.6 ± 1.3 | 13.9 ± 1.6 | 14.6 ± 1.3 |
| CLDM | 7.0M | 11.3 ± 1.0 | 14.3 ± 1.5 | 14.7 ± 1.4 |

**Vision and fusion on EgoEMG** (identical center frames; Frz./FT =
frozen/fine-tuned; Δavg = fusion gain):

| Backbone | Update | Vision Avg | Fusion Avg | Δavg |
|----------|--------|-----------|-----------|------|
| ResNet-18 | FT | 5.85 \* | 5.41 \* | +0.44 |
| ViT-S/14 | FT | 6.04 \* | 5.56 \* | +0.48 |
| ResNet-50 | Frz. | 5.27 | 5.19 | +0.09 |
| ResNet-152 | Frz. | 5.11 | 5.06 | +0.05 |
| ViT-B/14 | Frz. | 5.78 | 5.75 | +0.03 |
| ViT-L/14 | Frz. | 5.39 | 5.36 | +0.03 |
| WiLoR | Frz. | 4.73 | 4.68 | +0.04 |

</details>

### Evaluation

S/M/L in this README map to the `small`/`middle`/`large` config and
checkpoint variants (e.g. `EMGFormer-M` → `egoemg_emgformer_middle.ckpt`).

> **⚠️ EMG channel counts are NOT uniform across the released checkpoints.** A
> config and its checkpoint must come from the **same family**. The **EgoEMG**
> EMGFormer S/M/L checkpoints are **8-channel** (single-hand `target_hand`
> layout, `tds_slim` featurizer); the **EMG2Pose** EMGFormer S/M/L and both
> fusion checkpoints are **16-channel** (bilateral `emg2pose_interpolate16`
> layout, `tds_slim_16ch` featurizer). Pairing a 16-channel checkpoint with an
> 8-channel config (or vice-versa) fails with a featurizer shape mismatch —
> `egoemg_*` and `emg2pose_*` checkpoints are **not interchangeable** even
> though both use `_small`/`_middle`/`_large` names.

```shell
export EGOEMG_ROOT=/absolute/path/to/dataset_root

# EgoEMG EMGFormer-M (per-group stats + overall)
python -m egoemg.test_analysis \
  experiment=emgformer/egoemg_emgformer_middle \
  'checkpoint=checkpoints/egoemg_emgformer_middle.ckpt'

# EMG2Pose benchmark (EMGFormer-S)
python -m egoemg.test_analysis \
  experiment=emgformer/emg2pose_emgformer_small \
  'checkpoint=checkpoints/emg2pose_emgformer_small.ckpt' \
  data_location=${EGOEMG_ROOT}/data/emg2pose_memmap

# Vision-only ResNet-18
python -m egoemg.test_analysis \
  experiment=fusion/vision_resnet18 \
  'checkpoint=checkpoints/vision_resnet18.ckpt'

# Vision-only ViT-S/14
python -m egoemg.test_analysis \
  experiment=fusion/vision_vit_small \
  'checkpoint=checkpoints/vision_vit_small.ckpt'

# Fusion (ResNet-18 + 16ch EMG, WL 7790)
python -m egoemg.test_analysis \
  experiment=fusion/fusion_rn18_s_center_16ch_wl7790 \
  'checkpoint=checkpoints/fusion_resnet18_emgfusion_center.ckpt'

# Fusion (ViT-S/14 + 16ch EMG, WL 7790)
python -m egoemg.test_analysis \
  experiment=fusion/fusion_vits_s_center_eval_released \
  'checkpoint=checkpoints/fusion_vit_emgfusion_center.ckpt'
```

## 🗂️ Repository Layout

```text
egoemg/        models, datasets, training/eval, vendored UmeTrack FK
config/        Hydra experiments ({emgformer,fusion,emg2pose}/ + lineage/)
scripts/       data conversion, downloads, visualization, paper figures
experiments/   shell launchers for the paper's experiments
docs/          asset setup, support scope, config & fusion architecture
assets/        EMG layout figures and normalization statistics
```

## 📜 License

Code: **MIT**. Third-party material (incl. UmeTrack, CC-BY-NC-4.0) under
their own terms — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 📖 Citing EgoEMG

The paper is under review. Until a public version is available, cite this
repository and the exact commit — GitHub renders
[CITATION.cff](CITATION.cff).

<details>
<summary>Copyable BibTeX</summary>

```bibtex
@misc{egoemg2026,
  title        = {EgoEMG: A multimodal egocentric dataset with bilateral surface
                  EMG and webcam vision for hand pose estimation},
  author       = {Zhenqi Shi and others},
  howpublished = {GitHub repository},
  note         = {https://github.com/zhenqis123/EgoEMG},
  year         = {2026},
}
```

> The preprint entry and full author list will be published here once the
> paper is public.
</details>
