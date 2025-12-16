# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `emg2pose/`: data loading (`data.py`, `datasets/`), kinematics (`kinematics.py`, `UmeTrack/`), models and Lightning modules (`models/`, `modules.py`, `lightning.py`), training entrypoint (`train.py`), and utilities (`utils.py`, `transforms.py`, `visualization.py`).
- Hydra configs in `config/` (`experiment/`, `module/`, `optimizer/`, `lr_scheduler/`, `transforms/`, `data_split/`, `datamodule/`) drive experiments; override with `key=value` CLI flags.
- Tests live in `emg2pose/tests/` with fixtures in `assets/` and helper scripts in `scripts/`.
- Additional assets: `data/` (local downloads), `notebooks/` for exploratory workflows, `emg2pose_model_checkpoints/` for pretrained weights, and `docs/` for reference material.

## Build, Test, and Development Commands
- Set up environment and editable install: `conda env create -f environment.yml && conda activate emg2pose && pip install -e . && pip install -e emg2pose/UmeTrack`.
- Sanity-check training on the mini dataset:  
  `python -m emg2pose.train train=True eval=True experiment=tracking_vemg2pose trainer.max_epochs=5 data_split=mini_split data_location=$HOME/emg2pose_dataset_mini`
- Full training example:  
  `python -m emg2pose.train train=True eval=True experiment=tracking_vemg2pose data_location=$HOME/emg2pose_dataset`
- Evaluate a checkpoint only:  
  `python -m emg2pose.train train=False eval=True experiment=tracking_vemg2pose checkpoint=$HOME/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt data_location=$HOME/emg2pose_dataset`
- Run analysis suite: `python -m emg2pose.test_analysis experiment=tracking_vemg2pose checkpoint=... data_location=...`
- Run unit tests: `pytest emg2pose/tests -q`; add `-k name` to target a specific module.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, 88-character line limit (`flake8`, `isort`).
- Prefer type hints; `mypy` is configured with strict options (check untyped defs, warn on unused ignores).
- Imports are isort-formatted (multi-line mode 3, trailing commas); run `isort . && flake8` before pushing.
- Use snake_case for variables/functions, PascalCase for classes, and descriptive config names aligned with Hydra group folders.

## Testing Guidelines
- Use `pytest`; place new tests in `emg2pose/tests/` with filenames `test_*.py`.
- Keep tests deterministic and fast by relying on `tests/assets/` fixtures; avoid downloading large datasets in CI.
- For experiment changes, include a minimal CLI example in the test docstring or comments showing expected flags.
- Prefer asserting shapes/metrics from small tensors over full training loops; mock filesystem paths when possible.

## Commit & Pull Request Guidelines
- Commits should be small, imperative, and scoped (e.g., `Add mini split loader`); keep related changes together.
- Reference issues in PR descriptions, summarize intent and outcomes, and include CLI commands used for training/eval/tests.
- Attach logs or metric snippets when altering model behavior; note any new data dependencies or checkpoints.
- Ensure CLA compliance per `CONTRIBUTING.md`; include screenshots or tables for visualization/reporting changes when relevant.

## Configuration & Experiment Tips
- Start from `config/experiment/tracking_vemg2pose.yaml` or related experiments; override learning rate, batch size, or data paths via CLI (e.g., `optimizer.lr=1e-4 data_location=/path`).
- Keep custom configs under the existing Hydra groups to avoid path conflicts; commit only small, reproducible config deltas.
