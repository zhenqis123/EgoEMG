# EgoEMG

[ [`BibTeX`](#citing-egoemg) ]

A multimodal egocentric dataset with bilateral surface EMG and webcam vision
for hand pose estimation, together with EMG-to-pose, vision-to-pose, and
EMG+vision fusion baselines.

- **EMG-to-pose**: EMGFormer (small / middle / large) predicts hand pose from
  bilateral surface EMG.
- **Vision-to-pose**: ResNet / ViT single-frame predictors on the egocentric
  webcam stream.
- **EMG+vision fusion**: combines EMG and visual cues on center frames.

## Dataset

The EgoEMG dataset package (memmap data, all-intra webcam videos, pre-cropped
patches, metadata/calibration, and a visualization tool) is released under
CC-BY-NC-4.0 for research use and mirrored on Google Drive and Baidu Netdisk.

```shell
# Google Drive (default)
bash scripts/download/download_egoemg_data.sh

# or from Baidu Netdisk (requires `baidupcs` login)
bash scripts/download/download_egoemg_data.sh --source baidupcs
```

The `EgoEMG-dataset-small` package is a self-contained single-episode preview
(memmap + webcam video + pre-crop LMDB + metadata). See the package README for
the full layout. The complete unified memmap (EgoEMG + ShowEE + Incre), the
all-intra videos, and the pre-crop patches are distributed separately under
`EgoEMG_release/` on Baidu Netdisk / the corresponding Google Drive folders.

## Pre-trained Checkpoints

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
bash scripts/download/download_checkpoints.sh

# or from Baidu Netdisk (requires `baidupcs` login)
bash scripts/download/download_checkpoints.sh baidupcs
```

## Setup

```shell
conda env create -f environment.yml && conda activate emg2pose
pip install -e .
pip install -e egoemg/UmeTrack
```

Configs resolve data paths through `${oc.env:EMG2POSE_ROOT,./}`, so point it at
your dataset location:

```shell
export EMG2POSE_ROOT=/path/to/data_root
```

## Visualization

The dataset-centric visualization renders dataset-aligned frames for selected
samples (headless; writes PNG/MP4). It does not require the per-episode camera
calibration assets (it falls back to identity intrinsics when they are absent).

```shell
python scripts/viz/visualize_egoemg_vision_dataset.py \
  --memmap-dir ${EMG2POSE_ROOT}/data/EgoEMG_memmap \
  --video-root ${EMG2POSE_ROOT}/data/EgoEMG_allintra \
  --output-dir /tmp/egoemg_vision_viz \
  --num-samples 8 --target-hand both
```

## Reproducing Paper Results

Key numbers from the paper, reproduced by the provided checkpoints.

**EMG-to-pose on EgoEMG** (MAE in degrees, per-user mean ± std across the
Gesture / User / Both test splits; Avg. is the mean MAE):

| Method | Params | Gesture | User | Both | Avg. |
|--------|--------|---------|------|------|------|
| EMGFormer-S | 3.5M | 12.3 ± 1.5 | 16.0 ± 0.6 | 16.3 ± 1.5 | 14.1 |
| EMGFormer-M | 6.6M | 11.7 ± 1.6 | 15.9 ± 0.6 | 16.4 ± 1.5 | 13.8 |
| EMGFormer-L | 16.3M | 11.9 ± 1.6 | 16.0 ± 0.8 | 16.4 ± 1.2 | 13.9 |
| vEMG2Pose | 6.0M | 15.0 ± 1.4 | 16.3 ± 1.7 | 17.3 ± 1.3 | 15.9 |
| NeuroPose | 6.4M | 15.8 ± 1.2 | 15.7 ± 1.3 | 16.3 ± 0.7 | 16.1 |

**Vision-only → EMG+vision fusion on EgoEMG** (MAE in degrees on identical
center frames; Frz./FT = frozen/fine-tuned visual predictor; Avg. is the mean,
Δavg the fusion gain):

| Backbone | Update | Vision Avg | Fusion Avg | Δavg |
|----------|--------|-----------|-----------|------|
| ResNet-18 | FT | 5.84 | 5.40 | +0.44 |
| ViT-S/14 | FT | 6.02 | 5.54 | +0.48 |
| ResNet-50 | Frz. | 5.27 | 5.19 | +0.09 |
| ResNet-152 | Frz. | 5.11 | 5.06 | +0.05 |
| ViT-B/14 | Frz. | 5.78 | 5.75 | +0.03 |
| ViT-L/14 | Frz. | 5.39 | 5.36 | +0.03 |
| WiLoR | Frz. | 4.73 | 4.68 | +0.04 |

Evaluate a released checkpoint with `test_analysis`:

```shell
# EgoEMG EMGFormer (8ch target_hand layout)
python -m egoemg.test_analysis \
  experiment=emgformer/egoemg_emgformer_small \
  'checkpoint=/path/to/egoemg_emgformer_small.ckpt' \
  egoemg_unified_memmap_dir=${EMG2POSE_ROOT}/data/EgoEMG_unified_memmap

# EMG2Pose benchmark EMGFormer (16ch)
python -m egoemg.test_analysis \
  experiment=emgformer/emg2pose_emgformer_small \
  'checkpoint=/path/to/emg2pose_emgformer_small.ckpt' \
  data_location=${EMG2POSE_ROOT}/data/emg_corpus/emg2pose_v3_memmap
```

For vision and fusion checkpoints, use `test_analysis_fusion.py` with the
corresponding `vision_*` / `fusion_*` experiment config.

## License

The baseline code is distributed under the **MIT License**, as found in the
LICENSE file. The **EgoEMG dataset** is released under **CC-BY-NC-4.0** for
research use. Portions of this codebase are derived from
[emg2pose](https://github.com/facebookresearch/emg2pose), distributed under
CC-BY-NC-SA-4.0.

Third-party assets remain subject to their original licenses (UmeTrack,
the MANO model, and pretrained vision backbones).

## Citing EgoEMG

If you use this benchmark or dataset in your research, please cite:

```bibtex
@article{egoemg2026,
  title={EgoEmg: A Multimodal Egocentric Dataset with Bilateral EMG and Vision for Hand Pose Estimation},
  author={Anonymous},
  year={2026}
}
```
