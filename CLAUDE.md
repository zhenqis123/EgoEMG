# CLAUDE.md

This file provides agent-facing guidance for the active repository workflows.

## Project Scope

The repository documentation should focus on two active lines:

- `EMG -> pose`: EMGFormer-based hand pose prediction from EMG.
- `vision -> pose`: EgoEMG/WiLoR hand pose and MANO supervision from head-view
  frames.

Do not present unrelated model families as primary workflows. If legacy files
remain in the tree, treat them as non-mainline unless the user explicitly asks
to work on them.

## Environment

```bash
conda env create -f environment.yml && conda activate emg2pose
pip install -e .
```

## Main Commands

### EMGFormer EMG-to-Pose

Experiments are launched via shell scripts in `experiments/`:

```bash
# Standard EgoEMG regression (middle model)
bash experiments/regression/egoemg_middle_standard.sh

# Window length ablation sweep
bash experiments/ablation/window_length_sweep.sh

# Model size comparison
bash experiments/ablation/model_size_comparison.sh

# Finetune from pretrain checkpoint
bash experiments/finetune/egoemg_from_pretrain_middle.sh
```

Or directly via Hydra CLI with base templates:

```bash
# EgoEMG regression (template: regression_egoemg)
python -m egoemg.train \
  experiment=emgformer/regression_egoemg \
  egoemg_unified_memmap_dir=/path/to/EgoEMG_unified_memmap \
  'trainer.devices=[0,1,2,3,4,5]' \
  +trainer.strategy=ddp \
  batch_size=500

# emg2pose_v3 regression (template: regression_emg2pose)
python -m egoemg.train \
  experiment=emgformer/regression_emg2pose \
  'trainer.devices=[0,1,2,3,4,5]' \
  batch_size=600

# Evaluation only
python -m egoemg.train \
  train=False eval=True \
  experiment=emgformer/regression_egoemg \
  egoemg_unified_memmap_dir=/path/to/EgoEMG_unified_memmap \
  'checkpoint="/path/to/checkpoint.ckpt"' \
  'trainer.devices=[0]' batch_size=50

# Offline analysis
python -m egoemg.test_analysis \
  experiment=emgformer/regression_egoemg \
  'checkpoint="/path/to/checkpoint.ckpt"' \
  egoemg_unified_memmap_dir=/path/to/EgoEMG_unified_memmap
```

Use 5 base templates under `config/experiment/emgformer/`:
- `regression_egoemg.yaml` — EgoEMG regression (most active)
- `regression_emg2pose.yaml` — emg2pose_v3 regression
- `finetune_egoemg.yaml` — finetune from pretrain checkpoint
- `pretrain_multitask.yaml` — multi-task pretraining

Experiment scripts live in `experiments/{regression,finetune,pretrain,ablation}/`.
Legacy configs are archived in `config/experiment/{emgformer,fusion}/_archive/`.

### EgoEMG/WiLoR Vision-to-Pose

```bash
# Build the sidecar index once
python scripts/data/build_egoemg_vision_index.py \
  --memmap-dir /path/to/EgoEMG_unified_memmap \
  --output-dir /path/to/EgoEMG_unified_memmap/vision_index

# Video replay with mesh/markers/bbox overlay (most intuitive; the default
# recommended mode — see `visualize_dataset.py --help` for all modes)
python scripts/viz/visualize_dataset.py vision \
  --memmap-dir /path/to/EgoEMG_unified_memmap \
  --allintra-root /path/to/EgoEMG_allintra \
  --episode-id episode_000000 --stride 10 --max-frames 300

# Train and evaluate fusion / vision-to-pose on EgoEMG (single entrypoint)
python -m egoemg.train \
  experiment=fusion/fusion_rn18_s_center_8ch \
  train=true eval=true trainer.devices=[0,1,2,3,4]

# Vision-only single-frame baseline
python -m egoemg.train \
  experiment=fusion/vision_resnet18 \
  train=true eval=true trainer.devices=[0]

# Evaluate a trained vision/fusion checkpoint (center-frame eval, NO training).
# Use test_analysis, NOT `egoemg.train` — fusion configs default train=true and
# the train entrypoint would start a multi-hundred-epoch training run.
python -m egoemg.test_analysis \
  experiment=fusion/vision_resnet18 \
  'checkpoint=logs/fusion/vision_resnet/version_9/checkpoints/last.ckpt' \
  'trainer.devices=[0]'
```

Evaluation notes:

- The config must match the checkpoint's training setup. The released
  vision/fusion checkpoints use 16 EMG channels (`emg2pose_interpolate16` +
  `tds_slim_16ch`) at WL=7790 → use `fusion_*_center_16ch_wl7790.yaml` or
  `vision_resnet18.yaml` / `vision_vit_small.yaml`. The legacy 8ch
  (`target_hand`, WL=12000) configs do NOT load those checkpoints (featurizer
  shape mismatch).
- Fusion configs enable `center_frame_eval`, which selects the center-frame
  evaluation path in `test_analysis` (EMG-only configs instead report
  per-group stats). All `config/experiment/fusion/*` configs carry the flag.
- Checkpoint filenames containing `=` break Hydra CLI overrides — symlink to a
  `=`-free path or set `checkpoint:` in a user config.
- `EMG2POSE_ROOT` must be an absolute path (Hydra changes cwd at runtime).
- `test_analysis` writes `results.csv` into the Hydra run directory
  (`logs/<date>/<time>_emg2pose/`), never the repo root.

Vision/fusion config architecture: experiments in `config/experiment/fusion/`
inherit `config/lineage/fusion.yaml`. See `docs/config_architecture.md` for the
three-layer structure.

### Unified EgoEMG + ShowEE + Incre memmap (dataset merge)

Three recording corpora can be physically merged into one `egoemg_v2_memmap`-
format directory so training loads a single dataset instead of a mixed
`ConcatDataset`. Per-source availability is preserved via a per-frame
`dataset_source_id` field:

- **EgoEMG** — full supervision (EMG + joint angles + wrist + vision),
  `dataset_source_id=0`.
- **ShowEE** — wrist angles unavailable (zero-filled,
  `wrist_angles_valid=false`); the wrist loss is masked for ShowEE rows,
  `dataset_source_id=1`.
- **Incre** — vision/mocap unavailable (stale/invalid flags); only right-hand
  EMG + finger joint angles supervised, `dataset_source_id=2`.

Build the unified memmap and its norm stats, then train with
`dataset=egoemg_unified_angle_regression` and `egoemg_unified_memmap_dir`
pointing at the merged directory (see the config header for details).
Validation/test automatically use EgoEMG-only rows (ShowEE/Incre are train-only
augmentations):

```bash
python scripts/data/merge_datasets_to_unified_memmap.py \
    --egoemg <egoemg_v2_memmap_dir> \
    --showee <showee_memmap_dir> \
    --incre  <egoemg_incre>/data_right_merged \
    --out    <unified_memmap_dir>          # needs ~229 GB free
python scripts/data/compute_unified_norm_stats.py \
    --input assets/per_dataset_norm_stats_repro_filtered_paper_alias.json \
    --output assets/per_dataset_norm_stats_unified.json

# Training entrypoint (datamodule reads the unified memmap)
python -m egoemg.train \
  experiment=emgformer/regression_egoemg \
  dataset=egoemg_unified_angle_regression \
  egoemg_unified_memmap_dir=/path/to/unified_memmap \
  'trainer.devices=[0,1,2,3,4,5]' '+trainer.strategy=ddp' batch_size=500
```

Note: `scripts/data/merge_egoemg_incre.py` is the legacy single-pair merge;
`merge_datasets_to_unified_memmap.py` is the current three-source path.

## Tests And Checks

```bash
pytest egoemg/tests -q
pytest egoemg/tests -q -k test_name
isort . && flake8
mypy egoemg/
```

For dataset or visualization changes, prefer small smoke tests with synthetic
fixtures or `index_limit`. Avoid running full training unless explicitly needed.

## Architecture

### EMGFormer Path

```text
EMG window (B, C, T)
  -> featurizer
  -> transformer decoder / EMGFormer backbone
  -> prediction head
  -> joint angles or supervised targets
```

Key files:

- `egoemg/train.py`: Hydra supervised training entrypoint.
- `egoemg/lightning.py`: `EmgPredictionModule` for supervised training,
  metrics, checkpoint loading, and fine-tuning behavior.
- `egoemg/models/modules/emgformer.py`: `Emg2PoseFormer`.
- `egoemg/models/modules/emgformer_pretrain.py`: EMGFormer pretraining
  backbone used by EMGFormer configs that explicitly target it.
- `egoemg/models/featurizers/tds.py`: TDS-style temporal EMG featurizers.
- `egoemg/models/decoders/transformer.py`: transformer decoder components.
- `config/experiment/emgformer/`: active EMGFormer experiment configs.

### Vision-to-Pose Path

```text
EgoEMG head-view frame
  -> all-intra decord frame read
  -> calibrated projection and hand bbox
  -> WiLoR crop and normalization
  -> WiLoR MANO prediction and losses
```

Key files:

- `egoemg/train.py`: single Hydra entrypoint for all training (EMG, vision,
  fusion) — `python -m egoemg.train experiment=<group>/<name>`.
- `egoemg/datasets/egoemg_memmap_dataset.py`: EgoEMG memmap dataset that
  serves both EMG-only and vision (pre-crops + all-intra videos) samples.
- `egoemg/models/modules/mid_fusion.py`: `MidFusionPoseFormer` (EMG+vision
  fusion, center_supervised / vision_only / emg_only modes).
- `egoemg/models/modules/{wilor_vit,vit_vision,resnet_vision}.py`:
  vision-only backbone modules used by fusion experiments.
- `egoemg/video_io.py`: all-intra path resolution and `decord` reader cache.
- `egoemg/mano.py`: local MANO utilities; datasets should not initialize
  MANO.
- `scripts/data/build_egoemg_vision_index.py`: one-time sidecar index builder.
- `scripts/viz/visualize_dataset.py`: unified dataset visualization (modes: vision/timeline/mesh/fk_vs_mano).
- `config/lineage/fusion.yaml`: L1 shared defaults for vision/fusion
  experiments (see `docs/config_architecture.md`).

Known dataset issues (placeholder IMU, unreproducible filters, missing
raw files, engineering leftovers) are tracked in
`docs/data_known_issues.md` — check it before assuming a field's data is
real.

## Dataset Notes

- EMGFormer work should use the active memmap datasets and datamodule configs
  selected by `config/experiment/emgformer/`.
- EgoEMG vision startup depends on `vision_index`; do not restore frame-level
  full scans in dataset initialization.
- EgoEMG vision frame reads must use all-intra videos and `decord` only. Do not
  add OpenCV or original-video fallback paths.
- Keep visualization scripts headless-server friendly. Prefer writing PNG/MP4
  outputs instead of requiring an interactive display.

## EgoEMG MANO Semantics

- EgoEMG generated MANO labels use one canonical MANO-right semantic for both
  hands.
- Decode right-hand labels directly as MANO-right.
- Decode left-hand labels as MANO-right and recover left-hand display by
  reflection at projection/alignment time.
- `EgoEmgVisionDataset` must not initialize or run MANO. Keep MANO
  initialization/forward in the vision model path.

## Code Style

- Python 3.10+, 88-character line limit, 4-space indentation.
- Use type hints for new code.
- Format imports with isort and keep flake8 clean.
- Use snake_case for functions/variables and PascalCase for classes.
- Keep docs and config examples aligned with the two active workflows.
