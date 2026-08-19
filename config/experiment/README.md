# Experiment configuration status

All configurations under this directory are retained as research records during
the `0.1.0rc1` code pre-release. They compose successfully, but they are not a
promise of portable public training or evaluation: many require unpublished
data, checkpoints, or separately licensed WiLoR/MANO assets.

Directories named `_archive` are historical snapshots. They are intentionally
excluded from the active-config composition test and from
`scripts/release/audit_portability.py`.

Before a recipe is documented as public, it must have a data/model manifest,
portable paths, an explicit external-asset acquisition procedure, and a smoke
test using only released inputs.

