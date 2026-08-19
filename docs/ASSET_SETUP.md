# Legacy asset setup

This repository is a code pre-release, while the **current** EgoEMG release is
still being prepared. The download routes below provide the earlier **legacy
release**. It is the supported asset set for the legacy workflows explicitly
listed in the repository README; it is not a claim that the forthcoming dataset
or every historical experiment is released.

## 1. Choose an asset root

Keep downloaded data outside the Git checkout and export an absolute path. Hydra
creates run directories, so relative paths are not reliable.

```shell
export EMG2POSE_ROOT=/absolute/path/to/egoemg_assets
export EGOEMG_CHECKPOINT_ROOT=/absolute/path/to/emg2pose/checkpoints
```

The second variable is optional: canonical fusion configs default to the
repository-local `checkpoints/` directory created by the checkpoint downloader.

## 2. Preview package

The legacy preview is a single self-contained episode with memmap data,
all-intra video, precomputed crop LMDBs, metadata, and calibration. It is
intended for layout inspection and visualization smoke checks, not training or
paper-level evaluation.

```shell
pip install gdown
bash scripts/download/download_egoemg_data.sh \
  --source gdrive "$EMG2POSE_ROOT/data/dataset_egoemg_preview"

# Or, after logging into baidupcs:
bash scripts/download/download_egoemg_data.sh --source baidupcs
```

Inspect the downloaded package's `manifest.json` and `README.txt` before using
it as a visualizer input. Do not assume that the preview has the complete
release directory layout below.

## 3. Complete legacy data layout

The README's canonical training, evaluation, and full-video visualization
workflows require the complete legacy asset tree. Download it from the legacy
Baidu Netdisk share:

<https://pan.baidu.com/s/1aG2e-mHJkmP4KiYtYRcReA> (extraction code: `8059`).

The top-level `README.txt` in that share maps remote folders to the local layout.
The canonical paths expected by the README are:

```text
$EMG2POSE_ROOT/
├── assets/
│   ├── per_dataset_norm_stats.json
│   └── per_dataset_norm_stats_repro_filtered_paper_alias.json
└── data/
    ├── EgoEMG_unified_memmap/       # EMG training/evaluation + labels
    ├── EgoEMG_allintra/             # full-video visualization
    ├── EgoEMG_v2_crops/             # per-episode precomputed crop LMDBs
    └── emg_corpus/
        ├── emg2pose_v3/             # EMG2Pose source data
        └── emg2pose_v3_memmap/      # EMG2Pose training/evaluation memmap
```

For Baidu's split EMG2Pose files, reassemble after download:

```shell
bash scripts/download/assemble_emg2pose_parts.sh \
  "$EMG2POSE_ROOT/data/emg_corpus/emg2pose_v3_memmap"
```

## 4. Checkpoints

Download the ten legacy checkpoints into the repository-local `checkpoints/`
directory, or set `EGOEMG_CHECKPOINT_ROOT` to another directory containing the
same filenames.

```shell
pip install gdown
bash scripts/download/download_checkpoints.sh

# Or, after logging into baidupcs:
bash scripts/download/download_checkpoints.sh baidupcs
```

The README's canonical workflows use these files:

| Workflow | Required checkpoint(s) |
| --- | --- |
| EgoEMG EMGFormer evaluation | `egoemg_emgformer_small.ckpt` |
| EMG2Pose EMGFormer evaluation | `emg2pose_emgformer_small.ckpt` |
| ResNet-18 fusion initialization | `vision_resnet18.ckpt`, `egoemg_emgformer_small.ckpt` |
| ResNet-18 fusion evaluation | `fusion_resnet_emgfusion_center.ckpt` |

The legacy bundle also includes middle/large EMGFormer, ViT-S vision, and ViT-S
fusion checkpoints. Historical WiLoR and other research-only configurations are
outside this setup contract.

## 5. MANO and headless visualization

The `vision` visualizer requires separately licensed MANO model files in
addition to the legacy data. Obtain them under their own license and either pass
`--mano-model-path /path/to/mano/models` or set `WILOR_PATH` so that
`$WILOR_PATH/mano_data/models` exists. A WiLoR checkpoint is **not** required
for the canonical ResNet-18 or visualization workflows.

For full videos, use the complete layout above. `vision` reads the existing
precomputed LMDB crops; it does not derive crops from bounding boxes at runtime.

## 6. Verify before a long run

- Run `python scripts/release/audit_portability.py` to identify research-only
  configs; canonical README configs should not require developer `logs/` or
  `test_results/` paths.
- Run `python scripts/viz/visualize_dataset.py vision --help` to check Python
  dependencies and CLI availability.
- Start with the preview package for a visualization smoke check, then use the
  complete legacy tree for training and benchmark evaluation.

