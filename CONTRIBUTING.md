# Contributing to emg2pose

Thanks for your interest in contributing! This project follows a small set of
conventions to keep the codebase consistent and reproducible.

## Development setup

```shell
conda env create -f environment.yml && conda activate emg2pose
pip install -e . && pip install -e emg2pose/UmeTrack
```

## Code style

- Python 3.10+, 4-space indentation, 88-character line limit (`flake8`, `isort`).
- Prefer type hints; `mypy` is configured with strict options.
- Imports are isort-formatted (multi-line mode 3, trailing commas).
- `snake_case` for variables/functions, `PascalCase` for classes.

## Hydra configuration

Configs use a three-layer structure: `config/base.yaml` (framework skeleton) →
`config/lineage/*.yaml` (per-line defaults) → `config/experiment/<lineage>/*.yaml`
(experiment deltas). See `docs/config_architecture.md`. New experiments should
inherit the appropriate lineage and express only deltas. Use portable
`${oc.env:VAR,default}` interpolation for any data/dependency paths — never
hardcode machine-specific absolute paths.

## Tests

Place tests in `emg2pose/tests/` with `test_*.py` filenames. Keep them
deterministic and fast (small fixtures or synthetic tensors; no large dataset
downloads).

```shell
pytest emg2pose/tests -q
```

## Pull requests

- Keep commits focused; write clear commit messages.
- For experiment or dataset changes, include a minimal CLI example in docs or
  test comments showing expected flags.
- Do not commit checkpoints, logs, per-sample prediction arrays, or other bulky
  run artifacts — these are covered by `.gitignore`.
