# Repository Guidelines

## Project Scope
- Keep repository guidance aligned with the two active lines:
  `EMG -> pose` with EMGFormer, and `vision -> pose` with EgoEMG/WiLoR.
- Do not add new primary workflow documentation for unrelated model families.
- Prefer updating `docs/egoemg_wilor_training.md` for EgoEMG vision/WiLoR
  details, and this file or `CLAUDE.md` for agent-facing workflow guidance.

## Project Structure & Module Organization
- Core package lives in `egoemg/`: data loading (`datasets/`,
  `datamodule.py`), EMGFormer models (`models/modules/emgformer.py`,
  `models/modules/emgformer_pretrain.py`), fusion/vision modules
  (`models/modules/mid_fusion.py`, `models/modules/wilor_vit.py`,
  `models/modules/vit_vision.py`, `models/modules/resnet_vision.py`), Lightning
  modules, and utilities.
- Hydra configs live in `config/` with a three-layer structure:
  `config/base.yaml` (L0 framework skeleton) → `config/lineage/{emgformer,
  fusion,classic}.yaml` (L1 per-main-line shared defaults) →
  `config/experiment/{emgformer,fusion,emg2pose}/*.yaml` (L2 experiments
  expressing only deltas). See `docs/config_architecture.md` for details.
  The active experiment areas are `config/experiment/emgformer/` (EMG→pose),
  `config/experiment/fusion/` (vision→pose / EMG+vision fusion), and
  `config/experiment/emg2pose/` (classic baselines).
- Tests live in `egoemg/tests/`; scripts live in `scripts/`; reference docs
  live in `docs/`.
- EgoEMG vision data uses memmaps plus all-intra webcam videos. The sidecar
  vision index is required for fast startup.

## Build, Test, and Development Commands
- Set up environment and editable install:
  `conda env create -f environment.yml && conda activate emg2pose && pip install -e . && pip install -e egoemg/UmeTrack`.
- EMGFormer supervised EMG-to-pose training (single entrypoint for all lines):
  `python -m egoemg.train experiment=emgformer/regression_egoemg train=true eval=true trainer.devices=[0]`.
- EMGFormer pretraining: `python -m egoemg.train_pretrain` (uses
  `config_name=pretrain`; experiment configs in `config/experiment/emgformer/`).
- EgoEMG vision/fusion training (same entrypoint, fusion experiment):
  `python -m egoemg.train experiment=fusion/fusion_rn18_s_center_8ch train=true eval=true trainer.devices=[0,1,2,3,4]`.
- Vision/fusion EVALUATION uses `egoemg.test_analysis` (never `egoemg.train`
  — fusion configs default train=true and would start training). Fusion configs
  carry `center_frame_eval: true`; released vision/fusion checkpoints are
  16ch + WL=7790 → use `fusion_*_center_16ch_wl7790.yaml` / `vision_*` configs.
  See CLAUDE.md for examples and pitfalls.
- Build the EgoEMG vision sidecar index once:
  `python scripts/data/build_egoemg_vision_index.py --memmap-dir /path/to/EgoEMG_unified_memmap --output-dir /path/to/EgoEMG_unified_memmap/vision_index`.
- Visualize actual EgoEMG vision dataset samples:
  `python scripts/viz/visualize_egoemg_vision_dataset.py --memmap-dir /path/to/EgoEMG_unified_memmap --video-root /path/to/EgoEMG --allintra-root /path/to/EgoEMG_allintra --vision-index-dir /path/to/EgoEMG_unified_memmap/vision_index --output-dir /tmp/egoemg_vision_dataset_viz --num-samples 16 --target-hand both`.
- Merge EgoEMG + ShowEE + Incre into one unified memmap, then train with
  `dataset=egoemg_unified_angle_regression`:
  `python scripts/data/merge_datasets_to_unified_memmap.py --egoemg <dir> --showee <dir> --incre <egoemg_incre>/data_right_merged --out <dir>` followed by
  `python scripts/data/compute_unified_norm_stats.py --input assets/per_dataset_norm_stats_repro_filtered_paper_alias.json --output assets/per_dataset_norm_stats_unified.json`.
- Run unit tests: `pytest egoemg/tests -q`; add `-k name` to target a
  specific module.
- Validate Hydra config composition (no training): see
  `scripts/migrate/compare_resolved.py` (snapshot / verify-one / diff).

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, 88-character line limit (`flake8`,
  `isort`).
- Prefer type hints; `mypy` is configured with strict options.
- Imports are isort-formatted with multi-line mode 3 and trailing commas.
- Use snake_case for variables/functions, PascalCase for classes, and
  descriptive Hydra config names aligned with group folders.

## Testing Guidelines
- Place new tests in `egoemg/tests/` with filenames `test_*.py`.
- Keep tests deterministic and fast by relying on small fixtures or synthetic
  tensors. Avoid large dataset downloads in tests.
- For experiment or dataset changes, include a minimal CLI example in docs or
  test comments showing expected flags.
- Prefer asserting shapes, masks, projected coordinates, and key metrics over
  running full training loops.

## Configuration & Experiment Tips
- EMG-to-pose work should start from `config/experiment/emgformer/`
  (inherits `config/lineage/emgformer.yaml`).
- Vision/fusion-to-pose work should start from `config/experiment/fusion/`
  (inherits `config/lineage/fusion.yaml`).
- Classic baselines (emg2pose / neuropose / vemg2pose) live in
  `config/experiment/emg2pose/` (inherit `config/lineage/emg2pose.yaml`).
- New experiments should inherit the appropriate lineage and express only
  deltas. See `docs/config_architecture.md` for the layering rules.
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
