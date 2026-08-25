# Support scope

EgoEMG `0.1.0` is a code release. This page records what the release supports
and where the boundary lies.

## Supported

- Installing the Python package and running its data-independent tests.
- The README training, evaluation, and visualization commands, given the
  corresponding assets from `docs/ASSET_SETUP.md`.
- The Hydra configuration structure documented in
  `docs/config_architecture.md`.

## Outside the support surface

- Experiment configs that reference developer-local `logs/`, `test_results/`,
  `../WiLoR`, `../manotorch`, or unpublished paths. They are research records,
  not portable recipes; `python scripts/release/audit_portability.py` lists
  them.
- Historical workflows beyond the canonical asset layout.

## Visualization contract

`scripts/viz/visualize_dataset.py vision` emits an overlay MP4 and one MP4 per
hand. The hand videos are read exclusively from the precomputed episode LMDB;
they are never generated from an overlay bounding box at runtime. Frames whose
crops were never produced (no valid markers at capture time — e.g. frame 0 of
every episode) are skipped from the selection with a notice. For all retained
frames, output creation happens only after both crop keys are verified; a
missing LMDB or crop key is an error, not a black placeholder frame.

`--stride` and `--max-frames` select the same frame sequence for all three
videos, so the overlay and crop outputs have matching frame counts. To
produce videos spanning every unique source frame, use the default
`--stride 1` and omit `--max-frames`.

## External dependencies

The `viz` extra installs Python dependencies, but MANO model files and some
WiLoR assets are separately licensed and are not bundled. Headless rendering
uses EGL when available and otherwise depends on a working software-rendering
backend. See `docs/egoemg_wilor_training.md` before attempting a run on
private data.
