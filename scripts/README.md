# Scripts Overview

This directory contains reusable utilities organized by purpose.

## Directory Structure

| Dir | Purpose |
|---|---|
| `data/` | Dataset conversion (zarr↔memmap↔hdf5), download helpers, index builders |
| `prepare/` | Dataset preprocessing: crops, ViT features, IK masks, video encoding, marker fixing, data verification |
| `mano/` | MANO hand model inference, world transforms, alignment verification |
| `eval/` | Evaluation, per-gesture breakdown, fusion analysis, test analysis |
| `hparam/` | Optuna hyperparameter search (augmentation, window length) |
| `run/` | Experiment launcher shell scripts for active training and fusion workflows |
| `viz/` | Visualization scripts and sample outputs |
| `paper/` | Paper figure generation and appendix |
| `ik/` | Inverse kinematics mesh fitting (batch + single) |
| `util/` | Miscellaneous utilities (checkpoint surgery, temporary scripts) |

## Policy

- Each script lives in exactly one category subdirectory.
- Top-level `scripts/` contains only this README.
- Cross-references between scripts use paths relative to repo root (e.g. `python scripts/eval/analyze_per_gesture.py`).
