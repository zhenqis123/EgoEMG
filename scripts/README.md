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

## Support surface (`0.1.0rc1` code pre-release)

Only the entrypoints below are part of the public support surface
(documented, smoke-tested, expected to work with caller-owned assets).
Everything else is retained as a research record: it may hardcode
unpublished data locations (`logs/`, `test_results/`, `../WiLoR`) and is not
maintained for public use. The live list of non-portable references is
produced by `scripts/release/audit_portability.py`.

| Script | Purpose | Documented in |
|---|---|---|
| `viz/visualize_dataset.py` | Unified dataset visualizer (`vision`, `timeline`, `mesh`, `fk_vs_mano`) | README, `docs/PRERELEASE_LIMITATIONS.md` |
| `data/build_egoemg_vision_index.py` | Build the EgoEMG vision sidecar index | README, `docs/ASSET_SETUP.md` |
| `prepare/reencode_egoemg_webcam_allintra.py` | Re-encode webcam videos to all-intra for random access | `docs/ASSET_SETUP.md` |
| `release/audit_portability.py` | List non-portable references in active configs | `docs/PRERELEASE_LIMITATIONS.md` |
| `prepare/fix_egoemg_imu_channel_order.py` | One-shot data repair: reorder EgoEMG `imu` rows to `[acc, gyro]` (already applied to the unified memmap; kept for other copies and provenance) | `docs/data_known_issues.md` |
| `prepare/verify_original_lerobot_imu.py` | Verify IMU completeness/layout of the original LeRobot parquet data | `docs/data_known_issues.md` |

Promoting a research script into this table requires portable paths, an
asset manifest, and a data-independent smoke test (`docs/RELEASE_PLAN.md` §4.1).

## Policy

- Each script lives in exactly one category subdirectory.
- Top-level `scripts/` contains only this README.
- Cross-references between scripts use paths relative to repo root (e.g. `python scripts/eval/analyze_per_gesture.py`).
