# Changelog

All notable changes are recorded in this file.

## 0.1.0rc1 — unreleased code pre-release

- Declares the repository as a code pre-release; the current dataset release is
  not ready, while the earlier legacy data/checkpoint release remains supported
  for the canonical workflows documented in `docs/ASSET_SETUP.md`.
- Adds package metadata, optional dependency groups, source-distribution rules,
  and a wheel installation smoke check.
- Documents release limitations, data-card requirements, and third-party notices.
- Makes `vision` fail clearly when precomputed crop LMDB data is unavailable or
  incomplete, rather than silently encoding placeholder crop frames.
- Data repair: the EgoEMG-source rows of the unified-memmap `imu` field were
  reordered from the source parquet's gyro-first layout to the documented
  `[acc, gyro]` layout (41 episodes / 66,161,725 rows; backup
  `imu.dat.bak_prelayout`; verified bitwise against the original LeRobot
  parquet). The unified memmap's `imu_semantics` manifest entry documents the
  repair. See `docs/data_known_issues.md` #1 and
  `scripts/prepare/fix_egoemg_imu_channel_order.py`.
- Adds `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue templates, and a PR template;
  adds a secret-scan step and CLI `--help` smoke tests to CI; declares the
  scripts support surface in `scripts/README.md`.
- Records the original-data IMU verification report at
  `scripts/release/imu_verify_report_windows_original.json`.
- Static-analysis cleanup: removes unused imports and unused locals across the
  package, declares the visualization re-export surface with `__all__`, and
  extends CI lint to fail on unused-import/unused-variable regressions in the
  maintained package (`scripts/` stays fatal-errors-only as research records).
- Dependency-declaration audit: `joblib` (core `egoemg.utils`) and `omegaconf`
  were imported directly but never declared; both are now in
  `install_requires`. Remaining undeclared third-party imports (`av`, `timm`,
  `zarr`, `open3d`, `unidecode`) are lazy, guarded, or behind the lazy
  dataset factory and belong to optional research paths.
- Clean-runner hardening (all exposed by CI on a fresh install, all fixed):
  the built wheel was missing `egoemg.models` and its subpackages (namespace
  directories without `__init__.py` that `find_packages()` skips); an orphan
  empty `egoemg/utils/` directory shadowed the real `egoemg/utils.py` module;
  `egoemg.tests` needed a package marker for cross-test imports after the
  wheel force-reinstall; the classic visualization path needs PyAV via the
  vendored UmeTrack tracker (`av>=11` added to the `viz` extra); and the
  vendored UmeTrack tree's repo-root-absolute imports (`import lib...`) now
  resolve via a sys.path registration in `egoemg/UmeTrack/__init__.py`
  instead of relying on a legacy separate editable install. CI installs
  `.[dev,vision,viz]`, verifies core submodule imports from the wheel, and
  reports pytest FAILED/ERROR lines as annotations.
