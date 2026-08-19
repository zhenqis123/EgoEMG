# Code pre-release support boundary

EgoEMG `0.1.0rc1` is a code pre-release. This document defines the supported
surface until the data and model release is ready.

## Supported

- Installing the Python package and running its data-independent tests.
- Inspecting source code, Hydra configuration structure, and public command help.
- Running the visualizer only when the caller already has compatible memmap,
  all-intra video, calibration, MANO assets, and precomputed crop LMDB files.

## Not released or supported

- The current EgoEMG dataset release and any claim that it is identical to the
  legacy assets.
- Historical workflows outside the canonical legacy setup documented in
  `docs/ASSET_SETUP.md`.
- Experiment configs that reference developer-local `logs/`, `test_results/`,
  `../WiLoR`, `../manotorch`, or unpublished paths. They are research records,
  not portable release recipes.

Run `python scripts/release/audit_portability.py` to list these references in
the active config tree. Canonical README recipes use the legacy asset layout in
`docs/ASSET_SETUP.md`; the remaining findings are research-only configurations.

## Visualization contract

`scripts/viz/visualize_dataset.py vision` emits an overlay MP4 and one MP4 per
hand. The hand videos are read exclusively from the precomputed episode LMDB;
they are never generated from an overlay bounding box at runtime. Before output
creation, the command verifies that every selected video frame has both crop
keys. A missing LMDB or crop key is an error, not a black placeholder frame.

`--stride` and `--max-frames` select the same frame sequence for all three
videos. Therefore the overlay and crop outputs have matching frame counts. To
produce videos spanning every unique source frame, use the default `--stride 1`
and omit `--max-frames`.

## External dependencies

The `viz` extra installs Python dependencies, but MANO model files and some
WiLoR assets are separately licensed and are not bundled. Headless rendering
uses EGL when available and otherwise depends on a working software-rendering
backend. See the visualizer's error output and `docs/egoemg_wilor_training.md`
before attempting a private-data run.
