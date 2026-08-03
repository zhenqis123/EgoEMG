# emg2pose

[ [`Paper`](https://arxiv.org/abs/2412.02725) ] [ [`Dataset`](https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar) ] [ [`Blog`](https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/) ] [ [`BibTeX`](#citing-emg2pose) ]

A dataset of Surface electromyography (sEMG) recordings paired with ground-truth, motion-capture recordings of the hands. Data loading, baseline model training, and baseline model evaluation code are provided.

<p align="center">
  <img src="https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_overview.png" alt="EMG2Pose Overview" width="75%">
</p>


## Data
The entire dataset has $25,253$ HDF5 files, each consisting of time-aligned, 2kHz sEMG and joint angles for a single hand in a single stage. Each stage is ~1 minute. There are $193$ participants, spanning $370$ hours and $29$ stages. `egoemg.datasets.emg2pose_dataset.Emg2PoseSessionData` offers a programmatic read-only interface into the HDF5 session files.

The full dataset statistics are as follows:

<p align="center">
  <img src="images/dataset_stats.png" alt="Dataset statistics" width="75%">
</p>

The `metadata.csv` file includes the following information for each HDF5 file:

| Column             | Description |
|--------------------|-------------|
| `user`              | Anonymized user ID |
| `session`           | Recording session (there are multiple stages per recording session) |
| `stage`             | Name of stage |
| `side`              | Hand side (`left` or `right`) |
| `moving_hand`       | Whether the hand is prompted to move during the stage |
| `held_out_user`     | Whether the user is held out from the training set |
| `held_out_stage`    | Whether the stage is held out from the training set |
| `split`             | `train`, `test`, or `val` |
| `generalization`    | Type of generalization; across user (`user`), stage (`stage`), or across user and stage (`user_stage`) |

## Setup

### Environment and Dependencies

```shell
# Clone the repo, setup environment, and install local package
git clone https://github.com/<your-org>/emg2pose.git ~/emg2pose
cd ~/emg2pose
conda env create -f environment.yml

# Activate the environment
conda activate emg2pose

# Install the emg2pose package
pip install -e .

# Install the UmeTrack package (for forward kinematics and mesh skinning)
pip install -e egoemg/UmeTrack
```

### Data and External Dependency Paths

Configs use relative paths by default (data under `./data/`, the WiLoR
dependency at `../WiLoR`) but resolve them through Hydra `${oc.env:VAR,default}`
interpolation, so you can point them at your own locations via environment
variables:

```shell
# Repository / data root (defaults to the current directory).
export EMG2POSE_ROOT=/path/to/emg2pose
# WiLoR dependency checkout (defaults to ../WiLoR).
export WILOR_PATH=/path/to/WiLoR
# Optional: EMG corpus root used by stats scripts (defaults to ./data/emg_corpus).
export EMG_CORPUS_ROOT=/path/to/emg_corpus
```

See `docs/egoemg_wilor_training.md` for the full EgoEMG data layout and how to
prepare the memmap dataset, all-intra videos, and vision index.

### Unified EgoEMG + ShowEE + Incre memmap (recommended for training)

The three recording corpora can be physically merged into a single
`egoemg_v2_memmap` so training loads one dataset instead of a mixed
`ConcatDataset`. Per-source availability is preserved via a per-frame
`dataset_source_id` field:

- **EgoEMG** — full supervision (EMG + joint angles + wrist + vision).
- **ShowEE** — wrist angles are **unavailable** (zero-filled,
  `wrist_angles_valid=false`); the loss masks wrist channels for ShowEE rows.
- **Incre** — vision/mocap are **unavailable** (stale/invalid flags); only
  right-hand EMG + finger joint angles are supervised.

Build it with:

```shell
python scripts/data/merge_datasets_to_unified_memmap.py \
    --egoemg <egoemg_v2_memmap_dir> \
    --showee <showee_memmap_dir> \
    --incre  <egoemg_incre>/data_right_merged \
    --out    <unified_memmap_dir>          # needs ~229 GB free
python scripts/data/compute_unified_norm_stats.py \
    --input assets/per_dataset_norm_stats_repro_filtered_paper_alias.json \
    --output assets/per_dataset_norm_stats_unified.json
```

Then train with `dataset=egoemg_unified_angle_regression` and point
`egoemg_unified_memmap_dir` at the merged directory (see the config header for
details). Validation/test automatically use EgoEMG-only rows (ShowEE/Incre are
train-only augmentations).

## Getting Started (Small, Sanity-Check Dataset)


The full dataset is $431$ GiB -- which can be cumbersome for a quick start. As a solution, we
also host a smaller (~ $600$ MiB) version of the dataset which can be downloaded and used to run
a sanity-check version of the train and eval logic.

### (Optional) Download Just the Metadata CSV (5 MiB)

The `emg2pose_metadata.csv` file described above can be downloaded on its own using the following endpoint.

NOTE: this metadata file is also included in each of the dataset downloads

```shell
# Download (just) the metadata.csv file to ~/emg2pose_metadata.csv
cd ~ && curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv -o emg2pose_metadata.csv
```

### Download a Smaller Version of the Dataset (~600 MiB)

```shell
# Download a mini (600 MiB) version of the dataset
cd ~ && curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar" -o emg2pose_dataset_mini.tar

# Unpack the tar to ~/emg2pose_dataset_mini
tar -xvf emg2pose_dataset_mini.tar
```

### Sanity Check Train / Eval

To run a sanity-check training workflow over the small, sanity-check version of the
dataset, please use the following command.

This runs training for the `tracking_vemg2pose` experiment for $5$ epochs as a sanity check.
It also runs evaluation on the validation and test splits -- again as a sanity check.

```shell
python -m egoemg.train \
train=True \
eval=True \
experiment=tracking_vemg2pose \
trainer.max_epochs=5 \
data_split=mini_split \
data_location="${HOME}/emg2pose_dataset_mini"
```

## Getting Started (Full Dataset)

Above, we provided instructions for working with a smaller version of the dataset as a means
of sanity checking the main entrypoint (`train.py`). Here, we show how to get started with
the whole dataset.

### Download the Full Dataset (431 GiB)

```shell
# Download the full (431 GiB) version of the dataset, extract to ~/emg2pose_dataset
cd ~ && curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar -o emg2pose_dataset.tar

# Unpack the tar to ~/emg2pose_dataset
tar -xvf emg2pose_dataset.tar
```

### Train on the Full Dataset

To launch an example, full training run for the `vemg2pose (tracking)` setting, use the following:

```shell
python -m egoemg.train \
train=True \
eval=True \
experiment=tracking_vemg2pose \
data_location="${HOME}/emg2pose_dataset"
```

The `experiment` CLI option supports the following experiments (see `config/experiment` files):
* `tracking_vemg2pose`
* `regression_vemg2pose`
* `regression_neuropose`

## Downloading Pre-trained Checkpoints

We provide six pretrained checkpoints covering the main benchmark tasks (see
[Reproducing Paper Results](#reproducing-paper-results) below). They are
mirrored on Google Drive and Baidu Netdisk.

```shell
# Google Drive (default)
bash scripts/download/download_checkpoints.sh

# or from Baidu Netdisk (requires `baidupcs` login)
bash scripts/download/download_checkpoints.sh baidupcs
```

The script fetches the six `.ckpt` files into `checkpoints/`.

## Downloading the EgoEMG Dataset

The EgoEMG dataset package (`EgoEMG-dataset-small`) is released under
CC-BY-NC-4.0 for research use and mirrored on Google Drive and Baidu Netdisk.

```shell
# Google Drive (default)
bash scripts/download/download_egoemg_data.sh

# or from Baidu Netdisk (requires `baidupcs` login)
bash scripts/download/download_egoemg_data.sh --source baidupcs
```

The package is self-contained: it includes the memmap data, a webcam all-intra
video, pre-cropped LMDB shards, metadata/calibration, and a visualization tool.
See the package `README.md` for the full layout.

## Reproducing Paper Results

Key numbers from the paper, reproduced by the provided checkpoints.

**EMG-to-pose on EgoEMG** (MAE in degrees, per-user mean ± std across the
Gesture / User / Both test splits; Avg. is the mean MAE):

| Method | Params | Gesture | User | Both | Avg. |
|--------|--------|---------|------|------|------|
| EMGFormer-S | 3.5M | 12.8 ± 1.4 | 15.6 ± 2.5 | 17.4 ± 1.2 | 14.7 |
| EMGFormer-M | 6.6M | 11.8 ± 1.6 | 15.6 ± 1.4 | 17.4 ± 0.9 | 14.2 |
| EMGFormer-L | 16.3M | 11.7 ± 1.5 | 15.7 ± 3.0 | 17.7 ± 1.1 | 14.2 |
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

Aggregation follows `test_analysis.py` / `test_analysis_fusion.py`:
per-user mean ± std across the Gesture / User / Both splits, with the Avg.
column the mean MAE over all test splits. Vision and fusion report MAE on the
center frame of the sliding window.

## Evaluation / Testing

To run basic evaluation for the validation / test splits, use the following:

Note that the `experiment` option to this script should match the checkpoint's experiment.

```shell
# Run train.py with train=False to isolate basic evaluation logic
python -m egoemg.train \
train=False \
eval=True \
data_location="${HOME}/emg2pose_dataset" \
experiment=tracking_vemg2pose \
checkpoint="${HOME}/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt"
```

To run analyses for different modes of generalization and to generate a `.csv` file with results, use
the following script.

Note that the `experiment` option to this script should match the checkpoint's experiment.

```shell
python -m egoemg.test_analysis \
data_location="${HOME}/emg2pose_dataset" \
experiment=tracking_vemg2pose \
checkpoint="${HOME}/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt"
```

## Visualization

A brief walkthrough of data loading, inference, and data visualization is
available via the training/eval entrypoints documented in the sections above.

For EgoEMG vision supervision and WiLoR fine-tuning, see
`docs/egoemg_wilor_training.md`. This covers the memmap dataset, all-intra
video decoding, sidecar vision index generation, dataset visualization, and the
vision/fusion training/evaluation flow (run via `python -m egoemg.train`
with a fusion or vision-only experiment config).

## Workspace Organization

For local workspace hygiene, curated evaluation outputs, and guidance on what
should remain versioned versus local-only, see
`docs/workspace_organization.md`.

## License

The baseline code is distributed under the **MIT License**, as found in the
LICENSE file. The **EgoEMG dataset** is released under **CC-BY-NC-4.0** for
research use. Portions of this codebase are derived from
[emg2pose](https://github.com/facebookresearch/emg2pose), distributed under
CC-BY-NC-SA-4.0.

Third-party assets remain subject to their original licenses:

- UmeTrack is licensed under Attribution-NonCommercial 4.0 International, as
  found in `egoemg/UmeTrack/LICENSE` and
  [GitHub](https://github.com/facebookresearch/UmeTrack/blob/main/LICENSE).
- The MANO model and pretrained vision backbones are subject to their own
  licenses.

## Citing EgoEMG

If you use this benchmark or dataset in your research, please cite:

```
@article{egoemg2026,
  title={EgoEmg: A Multimodal Egocentric Dataset with Bilateral EMG and Vision for Hand Pose Estimation},
  author={Anonymous},
  year={2026}
}
```
