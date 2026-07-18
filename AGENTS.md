# Repository Guidelines

## Project Scope
- Keep repository guidance aligned with the two active lines:
  `EMG -> pose` with EMGFormer, and `vision -> pose` with EgoEMG/WiLoR.
- Do not add new primary workflow documentation for unrelated model families.
- Prefer updating `docs/egoemg_wilor_training.md` for EgoEMG vision/WiLoR
  details, and this file or `CLAUDE.md` for agent-facing workflow guidance.

## Project Structure & Module Organization
- Core package lives in `emg2pose/`: data loading (`datasets/`,
  `datamodule.py`), EMGFormer models (`models/modules/emgformer.py`,
  `models/modules/emgformer_pretrain.py`), vision/WiLoR modules
  (`models/wilor_egoemg.py`, `models/vision2pose.py`), Lightning modules, and
  utilities.
- Hydra configs live in `config/`. The active experiment areas are
  `config/experiment/emgformer/` and `config/experiment/vision/`, with
  `config/vision_base.yaml` for EgoEMG WiLoR fine-tuning.
- Tests live in `emg2pose/tests/`; scripts live in `scripts/`; reference docs
  live in `docs/`.
- EgoEMG vision data uses memmaps plus all-intra webcam videos. The sidecar
  vision index is required for fast startup.

## Build, Test, and Development Commands
- Set up environment and editable install:
  `conda env create -f environment.yml && conda activate emg2pose && pip install -e . && pip install -e emg2pose/UmeTrack`.
- EMGFormer supervised EMG-to-pose training:
  `python -m emg2pose.train train=True eval=True experiment=emgformer/regression_emgformer_middle_standard data_location=/path/to/emg2pose_memmap`.
- EMGFormer pretraining or supervised pretraining configs, when used, should
  come from `config/experiment/emgformer/` and run through the matching
  EMGFormer entrypoint documented in the config.
- EgoEMG WiLoR vision-to-pose training:
  `python -m emg2pose.train_vision data_location=/path/to/EgoEMG_memmap video_root=/path/to/EgoEMG allintra_root=/path/to/EgoEMG_allintra vision_index_dir=/path/to/EgoEMG_memmap/vision_index train=True eval=True`.
- Build the EgoEMG vision sidecar index once:
  `python scripts/data/build_egoemg_vision_index.py --memmap-dir /path/to/EgoEMG_memmap --output-dir /path/to/EgoEMG_memmap/vision_index`.
- Visualize actual EgoEMG vision dataset samples:
  `python scripts/viz/visualize_egoemg_vision_dataset.py --memmap-dir /path/to/EgoEMG_memmap --video-root /path/to/EgoEMG --allintra-root /path/to/EgoEMG_allintra --vision-index-dir /path/to/EgoEMG_memmap/vision_index --output-dir /tmp/egoemg_vision_dataset_viz --num-samples 16 --target-hand both`.
- Run unit tests: `pytest emg2pose/tests -q`; add `-k name` to target a
  specific module.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, 88-character line limit (`flake8`,
  `isort`).
- Prefer type hints; `mypy` is configured with strict options.
- Imports are isort-formatted with multi-line mode 3 and trailing commas.
- Use snake_case for variables/functions, PascalCase for classes, and
  descriptive Hydra config names aligned with group folders.

## Testing Guidelines
- Place new tests in `emg2pose/tests/` with filenames `test_*.py`.
- Keep tests deterministic and fast by relying on small fixtures or synthetic
  tensors. Avoid large dataset downloads in tests.
- For experiment or dataset changes, include a minimal CLI example in docs or
  test comments showing expected flags.
- Prefer asserting shapes, masks, projected coordinates, and key metrics over
  running full training loops.

## Configuration & Experiment Tips
- EMG-to-pose work should start from `config/experiment/emgformer/`.
- Vision-to-pose work should start from `config/vision_base.yaml` or
  `config/experiment/vision/`.
- Keep custom configs under existing Hydra groups. Commit only small,
  reproducible config deltas.
- For EgoEMG vision, always use all-intra videos and `decord`; do not add a
  fallback to original long-GOP videos or OpenCV decoding.

## EgoEMG MANO Semantics
- EgoEMG generated MANO labels use a single canonical MANO-right semantic for
  both hands.
- `*_right_pose.npy` is decoded with MANO-right semantics directly.
- `*_left_pose.npy` must also be decoded with MANO-right semantics; left-hand
  visualization/reprojection is recovered by applying a left-hand reflection at
  display/alignment time.
- Do not interpret EgoEMG `*_left_pose.npy` as MANO-left pose semantics.
- `EgoEmgVisionDataset` must not initialize or run MANO. Keep MANO
  initialization/forward in the vision model path.
