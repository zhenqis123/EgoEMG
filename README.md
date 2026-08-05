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
  <a href="#-dataset"><img src="https://img.shields.io/badge/Dataset-CC--BY--NC--4.0-orange" alt="Dataset license"></a>
  <a href="#-license"><img src="https://img.shields.io/badge/Code-MIT-blue" alt="Code license"></a>
  <a href="#-pre-trained-checkpoints"><img src="https://img.shields.io/badge/checkpoints-10-brightgreen" alt="Checkpoints"></a>
  <a href="https://github.com/zhenqis123/EgoEMG/actions"><img src="https://img.shields.io/github/actions/workflow/status/zhenqis123/EgoEMG/main.yml?branch=main" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/version-1.0.0-8A2BE2" alt="Version">
</p>

<p align="center">
  [ <a href="#-reproducing-paper-results">Results</a> ·
    <a href="#-dataset">Dataset</a> ·
    <a href="#-pre-trained-checkpoints">Checkpoints</a> ·
    <a href="#-citing-egoemg">Citation</a> ]
</p>

---

## ✨ Highlights

- **🧠 EMG-to-pose** — EMGFormer (small / middle / large) predicts hand pose from
  bilateral surface EMG.
- **👁️ Vision-to-pose** — ResNet / ViT single-frame predictors on the egocentric
  webcam stream.
- **🔀 EMG+vision fusion** — combines EMG and visual cues on identical
  center frames.

## 📖 Contents

- [Dataset](#-dataset)
- [Pre-trained Checkpoints](#-pre-trained-checkpoints)
- [Setup](#-setup)
- [Training](#-training)
- [Visualization](#-visualization)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Repository Layout](#-repository-layout)
- [License](#-license)
- [Citation](#-citing-egoemg)

## 📦 Dataset

EgoEMG is released under **CC-BY-NC-4.0** for research use. It pairs hand-pose
labels with bilateral sEMG, wrist IMUs, egocentric RGB video, external RGB-D,
and motion-capture annotations.

### Preview package

The self-contained `dataset_egoemg_preview` package contains one episode of
memmap data, all-intra webcam video, pre-cropped LMDB patches, and
metadata/calibration. Download it for a quick data-layout inspection:

```shell
# Google Drive (default; install gdown once with: pip install gdown)
pip install gdown
bash scripts/download/download_egoemg_data.sh

# or from Baidu Netdisk (requires `baidupcs` login)
bash scripts/download/download_egoemg_data.sh --source baidupcs
```

### Full release

The EgoEMG EMGFormer, vision, and fusion workflows require the complete
unified memmap (`EgoEMG + ShowEE + Incre`). ResNet/ViT vision and fusion
experiments also use the pre-cropped image patches; the actual-frame/WiLoR
pipeline additionally needs all-intra webcam videos. These assets are
distributed separately under `EgoEMG_release/` on Baidu Netdisk and in the
corresponding Google Drive folders. EMG2Pose benchmark checkpoints instead
require the corresponding EMG2Pose memmap (see [Evaluation](#evaluation)).

Keep the full release under a data root and set `EMG2POSE_ROOT` to its
**absolute** path. The supplied configurations resolve assets relative to this
variable; Hydra run directories make relative paths unreliable.

```shell
export EMG2POSE_ROOT=/absolute/path/to/data_root
```

> **Baidu Netdisk note**: single files are capped at 128 GB there, so the two
> largest files of `dataset_emg2pose_benchmark` (`emg.dat`, `joint_angles.dat`)
> are distributed as 100 GB `*.part_XX` chunks — reassemble them with
> `bash scripts/download/assemble_emg2pose_parts.sh <memmap_dir>` after
> downloading (the Google Drive copies are already assembled).

**Manual download (Baidu Netdisk, permanent link):**
<https://pan.baidu.com/s/1aG2e-mHJkmP4KiYtYRcReA> — 提取码 `8059`.
This link hosts the entire `EgoEMG_release/` tree (unified memmap, all-intra
videos, pre-crop patches, and all checkpoints). Downloading via the `baidupcs`
CLI (`baidupcs download /EgoEMG_release/...`) does not need the code.
The root `README.txt` inside the share maps each remote directory to its
local `data/` layout (e.g. `dataset_egoemg_unified` → `data/EgoEMG_unified_memmap`).

## 🏆 Pre-trained Checkpoints

Ten pretrained checkpoints are provided and mirrored on Google Drive and Baidu
Netdisk:

| Checkpoint | Task |
|-----------|------|
| `egoemg_emgformer_small.ckpt` | EMG-to-pose on EgoEMG (EMGFormer-S) |
| `egoemg_emgformer_middle.ckpt` | EMG-to-pose on EgoEMG (EMGFormer-M) |
| `egoemg_emgformer_large.ckpt` | EMG-to-pose on EgoEMG (EMGFormer-L) |
| `emg2pose_emgformer_small.ckpt` | EMG-to-pose on EMG2Pose (EMGFormer-S) |
| `emg2pose_emgformer_middle.ckpt` | EMG-to-pose on EMG2Pose (EMGFormer-M) |
| `emg2pose_emgformer_large.ckpt` | EMG-to-pose on EMG2Pose (EMGFormer-L) |
| `vision_resnet18.ckpt` | Vision-to-pose (ResNet-18) |
| `vision_vit_small.ckpt` | Vision-to-pose (ViT-S) |
| `fusion_resnet_emgfusion_center.ckpt` | EMG+Vision fusion (ResNet-18) |
| `fusion_vit_emgfusion_center.ckpt` | EMG+Vision fusion (ViT-S) |

```shell
# Google Drive (default)
pip install gdown
bash scripts/download/download_checkpoints.sh

# or from Baidu Netdisk (requires `baidupcs` login)
bash scripts/download/download_checkpoints.sh baidupcs
```

The checkpoints are also included in the Baidu Netdisk share link above
(<https://pan.baidu.com/s/1aG2e-mHJkmP4KiYtYRcReA>, 提取码 `8059`).

## ⚙️ Setup

```shell
conda env create -f environment.yml && conda activate emg2pose
pip install -e .
```

The Google Drive download scripts use `gdown`, which is not included in the
training environment. Install it only when you use that download route.
Set `EMG2POSE_ROOT` as shown in [Full release](#full-release) before running a
configuration that reads dataset assets.

## 🚀 Training

Supervised EMG, vision, and fusion training use `egoemg.train`; the Hydra
experiment selects the model family. The commands below assume the full release
is available at `$EMG2POSE_ROOT`:

> **Hardware:** the reference recipes use NVIDIA GPUs with CUDA 11.8. The EMG
> example below is a six-GPU run and the fusion example uses five GPUs. For a
> one-GPU smoke run, set `trainer.devices=[0]` and reduce `batch_size` (and
> `val_batch_size` when applicable) to fit your GPU memory.

```shell
# EMG-to-pose (EMGFormer-M regression recipe on EgoEMG)
python -m egoemg.train \
  experiment=emgformer/regression_egoemg \
  'trainer.devices=[0,1,2,3,4,5]' '+trainer.strategy=ddp' \
  batch_size=500

# Vision-only single-frame baseline (ResNet-18)
python -m egoemg.train \
  experiment=fusion/vision_resnet18 \
  train=true eval=true 'trainer.devices=[0]'

# EMG+vision fusion (ResNet-18 + EMGFormer-S, center-supervised)
python -m egoemg.train \
  experiment=fusion/fusion_rn18_s_center_16ch_wl7790 \
  train=true eval=true 'trainer.devices=[0,1,2,3,4]'

# EMGFormer multi-task pretraining
python -m egoemg.train_pretrain \
  experiment=emgformer/pretrain_multitask \
  train=true eval=false 'trainer.devices=[0]'
```

Active experiments live in `config/experiment/emgformer/` and
`config/experiment/fusion/`; shell launchers for the paper experiments live in
`experiments/`. For evaluation, use `egoemg.test_analysis` (see
[Evaluation](#evaluation)): fusion configs default to `train=true`, so the
training entrypoint would otherwise start a new run.

## 🎥 Visualization

The dataset-centric visualizer renders the exact samples emitted by
`EgoEmgVisionDataset`, including the dataset-aligned frame, hand box,
projected labels, and normalized training patch. It is headless and writes one
PNG per selected sample.

```shell
python scripts/viz/visualize_egoemg_vision_dataset.py \
  --memmap-dir ${EMG2POSE_ROOT}/data/EgoEMG_unified_memmap \
  --video-root ${EMG2POSE_ROOT}/data/EgoEMG \
  --allintra-root ${EMG2POSE_ROOT}/data/EgoEMG_allintra \
  --output-dir /tmp/egoemg_vision_viz \
  --num-samples 8 --target-hand both \
  --auto-build-index   # builds <memmap-dir>/vision_index if missing
```

For a larger visual check or WiLoR-specific setup notes, see
[the EgoEMG/WiLoR training guide](docs/egoemg_wilor_training.md).

## 📊 Reproducing Paper Results

Key numbers from the paper. Rows backed by a released checkpoint (see
[Pre-trained Checkpoints](#-pre-trained-checkpoints)) are reproducible with the
commands below; the remaining rows are reported from the paper only.

**EMG-to-pose on EgoEMG** (MAE in degrees, per-user mean ± std across the
Gesture / User / Both test splits; Avg. is the per-sample-weighted MAE across
the three test splits):

| Method | Params | Gesture | User | Both | Avg. |
|--------|--------|---------|------|------|------|
| EMGFormer-S | 3.5M | 12.8 ± 1.4 | 15.6 ± 2.5 | 17.4 ± 1.2 | 14.7 |
| EMGFormer-M | 6.6M | 11.8 ± 1.6 | 15.6 ± 1.4 | 17.4 ± 0.9 | 14.2 |
| EMGFormer-L | 16.3M | 11.7 ± 1.5 | 15.7 ± 3.0 | 17.7 ± 1.1 | 14.2 |
| vEMG2Pose | 6.0M | 15.0 ± 1.4 | 16.3 ± 1.7 | 17.3 ± 1.3 | 15.9 |
| NeuroPose | 6.4M | 15.8 ± 1.2 | 15.7 ± 1.3 | 16.3 ± 0.7 | 16.1 |

*`vEMG2Pose` and `NeuroPose` rows are paper-reported; their checkpoints are not
released.*

**EMG-to-pose on the EMG2Pose benchmark** (MAE in degrees, per-user mean ± std
across the User / Stage / User+Stage test splits):

| Method | Params | User | Stage | User+Stage |
|--------|--------|------|-------|------------|
| EMGFormer-S | 3.5M | 12.5 ± 1.1 | 11.1 ± 1.2 | 12.3 ± 1.1 |
| EMGFormer-M | 6.6M | 12.4 ± 1.1 | 10.2 ± 1.1 | 12.4 ± 1.1 |
| EMGFormer-L | 16.3M | 12.3 ± 1.1 | 9.3 ± 1.1 | 12.3 ± 1.1 |

*Rows correspond to the released `emg2pose_emgformer_{small,middle,large}.ckpt`
checkpoints; prior-work rows in the paper are paper-reported.*

**Vision-only → EMG+vision fusion on EgoEMG** (MAE in degrees on identical
center frames; Frz./FT = frozen/fine-tuned visual predictor; Avg. is the
per-sample-weighted MAE, Δavg the fusion gain):

| Backbone | Update | Vision Avg | Fusion Avg | Δavg |
|----------|--------|-----------|-----------|------|
| ResNet-18 | FT | 5.84 | 5.40 | +0.44 |
| ViT-S/14 | FT | 6.02 | 5.54 | +0.48 |
| ResNet-50 | Frz. | 5.27 | 5.19 | +0.09 |
| ResNet-152 | Frz. | 5.11 | 5.06 | +0.05 |
| ViT-B/14 | Frz. | 5.78 | 5.75 | +0.03 |
| ViT-L/14 | Frz. | 5.39 | 5.36 | +0.03 |
| WiLoR | Frz. | 4.73 | 4.68 | +0.04 |

*Released checkpoints cover the fine-tuned ResNet-18 / ViT-S rows and their
fusions; the frozen-backbone rows (ResNet-50/152, ViT-B/L, WiLoR) are
paper-reported.*

### Evaluation

Evaluate a released checkpoint with `test_analysis`. First download the
checkpoints (see [Pre-trained Checkpoints](#-pre-trained-checkpoints); they
land in `checkpoints/`) and export `EMG2POSE_ROOT` as an **absolute** path.
The EgoEMG experiment configs enable `per_group_stats` by default, so the
output reports the per-user / per-gesture mean ± std (matching the paper
tables) plus an `overall` MAE: the same per-sample-weighted aggregate reported
in the paper's Avg column:

```shell
export EMG2POSE_ROOT=/absolute/path/to/dataset_root

# EgoEMG EMGFormer (8ch target_hand layout); reports per-group stats + overall
python -m egoemg.test_analysis \
  experiment=emgformer/egoemg_emgformer_small \
  'checkpoint=checkpoints/egoemg_emgformer_small.ckpt'

# EMG2Pose benchmark EMGFormer (16ch)
python -m egoemg.test_analysis \
  experiment=emgformer/emg2pose_emgformer_small \
  'checkpoint=checkpoints/emg2pose_emgformer_small.ckpt' \
  data_location=${EMG2POSE_ROOT}/data/emg_corpus/emg2pose_v3_memmap
```

**Vision and fusion** checkpoints use the same `test_analysis` entrypoint;
their experiment configs enable `center_frame_eval`, which evaluates on
identical center frames and reports the per-hand MAE. The config must match
the checkpoint's training setup (EMG channels, window length): the released
vision/fusion checkpoints were trained with 16 EMG channels
(`emg2pose_interpolate16` layout) at WL=7790, so use the `*_16ch_wl7790`
fusion configs:

```shell
# Vision-only ResNet18
python -m egoemg.test_analysis \
  experiment=fusion/vision_resnet18 \
  'checkpoint=checkpoints/vision_resnet18.ckpt'

# EMG+vision fusion (ResNet18 + EMGFormer-S)
python -m egoemg.test_analysis \
  experiment=fusion/fusion_rn18_s_center_16ch_wl7790 \
  'checkpoint=checkpoints/fusion_resnet_emgfusion_center.ckpt'
```

Notes for evaluation:

- Checkpoint filenames containing `=` (e.g. `epoch=011-val_mae=0.1022.ckpt`)
  cannot be passed through Hydra CLI overrides; symlink to a `=`-free path or
  set the `checkpoint:` field in a user config.
- Fusion and vision evals use the same 2096/2082 center frames per hand.
  If a fusion eval reports fewer samples, its evaluation window is longer
  than the reference WL=7790 grid and frames near episode edges are dropped
  (the model window must fit inside the episode). Expected, not a bug.
- `results.csv` is written into the Hydra run directory (`logs/<date>/...`),
  never into the repo root.

`test_analysis` is the single evaluation tool for all three modalities
(EMG generalization splits with per-group stats, and vision/fusion center-frame
evaluation), selected automatically by the experiment config.

## 🗂️ Repository Layout

- `egoemg/` — source code: models, dataset wrappers, training/eval entrypoints
  (`train.py`, `test_analysis.py`), and vendored `UmeTrack` FK utilities.
- `config/` — Hydra experiment configs (`config/experiment/{emgformer,fusion}/`)
  over shared lineage defaults (`config/lineage/`).
- `scripts/` — data conversion, dataset/checkpoint download, visualization,
  and paper-figure regeneration.
- `experiments/` — shell launchers for the paper's experiments.
- `docs/` — config architecture and dataset notes.
- `assets/` — EMG layout figures and per-dataset normalization statistics.

## 📜 License

The baseline code is distributed under the **MIT License**, as found in the
LICENSE file. The **EgoEMG dataset** is released under **CC-BY-NC-4.0** for
research use. Portions of this codebase are derived from
[emg2pose](https://github.com/facebookresearch/emg2pose), distributed under
CC-BY-NC-SA-4.0.

Third-party assets remain subject to their original licenses (UmeTrack,
the MANO model, and pretrained vision backbones).

## 📖 Citing EgoEMG

The paper is currently under review. Official citation metadata will be added
when a public version is available. Until then, please reference this
repository and the exact commit used for your experiments.
