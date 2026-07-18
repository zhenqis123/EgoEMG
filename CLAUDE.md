# CLAUDE.md

This file provides agent-facing guidance for the active repository workflows.
Keep it synchronized with `AGENTS.md`.

## Project Scope

The repository documentation should focus on two active lines:

- `EMG -> pose`: EMGFormer-based hand pose prediction from EMG.
- `vision -> pose`: EgoEMG/WiLoR hand pose and MANO supervision from webcam
  frames.

Do not present unrelated model families as primary workflows. If legacy files
remain in the tree, treat them as non-mainline unless the user explicitly asks
to work on them.

## Environment

```bash
conda env create -f environment.yml && conda activate emg2pose
pip install -e .
pip install -e emg2pose/UmeTrack
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
python -m emg2pose.train \
  experiment=emgformer/regression_egoemg \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap \
  'trainer.devices=[0,1,2,3,4,5]' \
  +trainer.strategy=ddp \
  batch_size=500

# emg2pose_v3 regression (template: regression_emg2pose)
python -m emg2pose.train \
  experiment=emgformer/regression_emg2pose \
  'trainer.devices=[0,1,2,3,4,5]' \
  batch_size=600

# Evaluation only
python -m emg2pose.train \
  train=False eval=True \
  experiment=emgformer/regression_egoemg \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap \
  'checkpoint="/path/to/checkpoint.ckpt"' \
  'trainer.devices=[0]' batch_size=50

# Offline analysis
python -m emg2pose.test_analysis \
  experiment=emgformer/regression_egoemg \
  'checkpoint="/path/to/checkpoint.ckpt"' \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap
```

Use 5 base templates under `config/experiment/emgformer/`:
- `regression_egoemg.yaml` — EgoEMG regression (most active)
- `regression_emg2pose.yaml` — emg2pose_v3 regression
- `finetune_egoemg.yaml` — finetune from pretrain checkpoint
- `pretrain_multitask.yaml` — multi-task pretraining

Experiment scripts live in `experiments/{regression,finetune,pretrain,ablation}/`.
Legacy configs are archived in `config/experiment/emgformer/_archive/`.

### EgoEMG/WiLoR Vision-to-Pose

```bash
# Build the sidecar index once
python scripts/data/build_egoemg_vision_index.py \
  --memmap-dir /path/to/EgoEMG_memmap \
  --output-dir /path/to/EgoEMG_memmap/vision_index

# Visualize actual dataset samples
python scripts/viz/visualize_egoemg_vision_dataset.py \
  --memmap-dir /path/to/EgoEMG_memmap \
  --video-root /path/to/EgoEMG \
  --allintra-root /path/to/EgoEMG_allintra \
  --vision-index-dir /path/to/EgoEMG_memmap/vision_index \
  --output-dir /tmp/egoemg_vision_dataset_viz \
  --num-samples 16 \
  --target-hand both

# Train and evaluate WiLoR on EgoEMG
python -m emg2pose.train_vision \
  data_location=/path/to/EgoEMG_memmap \
  video_root=/path/to/EgoEMG \
  allintra_root=/path/to/EgoEMG_allintra \
  vision_index_dir=/path/to/EgoEMG_memmap/vision_index \
  mano_model_path=/path/to/mano_data \
  wilor_checkpoint_path=/path/to/wilor_final.ckpt \
  train=True \
  eval=True

# Vision-only single-frame baseline (no RefineNet, direct ViT → joint angles)
python -m emg2pose.train_vision \
  experiment=vision/vision_only_angle
```

Full EgoEMG/WiLoR workflow details live in
`docs/egoemg_wilor_training.md`.

## Tests And Checks

```bash
pytest emg2pose/tests -q
pytest emg2pose/tests -q -k test_name
isort . && flake8
mypy emg2pose/
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

- `emg2pose/train.py`: Hydra supervised training entrypoint.
- `emg2pose/lightning.py`: `EmgPredictionModule` for supervised training,
  metrics, checkpoint loading, and fine-tuning behavior.
- `emg2pose/models/modules/emgformer.py`: `Emg2PoseFormer`.
- `emg2pose/models/modules/emgformer_pretrain.py`: EMGFormer pretraining
  backbone used by EMGFormer configs that explicitly target it.
- `emg2pose/models/featurizers/tds.py`: TDS-style temporal EMG featurizers.
- `emg2pose/models/decoders/transformer.py`: transformer decoder components.
- `config/experiment/emgformer/`: active EMGFormer experiment configs.

### Vision-to-Pose Path

```text
EgoEMG webcam frame
  -> all-intra decord frame read
  -> calibrated projection and hand bbox
  -> WiLoR crop and normalization
  -> WiLoR MANO prediction and losses
```

Key files:

- `emg2pose/train_vision.py`: Hydra entrypoint for EgoEMG/WiLoR training and
  evaluation.
- `emg2pose/datasets/egoemg_vision_dataset.py`: EgoEMG memmap + video dataset
  that emits WiLoR-native samples.
- `emg2pose/models/wilor_egoemg.py`: `EgoEMGWiLoRModule`.
- `emg2pose/video_io.py`: all-intra path resolution and `decord` reader cache.
- `emg2pose/mano.py`: local MANO utilities; datasets should not initialize
  MANO.
- `scripts/data/build_egoemg_vision_index.py`: one-time sidecar index builder.
- `scripts/viz/visualize_egoemg_vision_dataset.py`: dataset visualization debug.
- `config/vision_base.yaml`: base WiLoR fine-tuning config.

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
