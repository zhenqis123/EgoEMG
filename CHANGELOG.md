# Changelog

All notable changes are recorded in this file.

## 0.1.0 — first public release (2026-08-23)

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
- Review-driven correctness batch (independent five-agent audit, tracked in
  `docs/code_review_findings_20260820.md`): fixes the SDPA bool-mask semantic
  inversion in rotary attention (future leakage + NaN under `causal=true`),
  the multitask pretraining heads missing `_target_` and the 8-vs-16-channel
  featurizer mismatch, vision validity on missing precomputed crops, MixUp
  target/mask consistency, the ResNet `head_vision.` double-prefix strip,
  `weights_only` handling for checkpoint loads, center-frame robustness
  (name-based split resolution, CPU fallback, configurable EMG preference),
  keystroke collate/consumer None-safety, offline evaluation (no ImageNet
  downloads), pooled-evaluation pairing by key, dtype-preserving numpy
  augmentations, and early validation of missing EMG variants. Center-frame
  evaluation keeps including missing-crop black frames by maintainer decision.
- Documents the legacy-release IMU channel-order fix for downloaders
  (checksums and self-patch command in `docs/ASSET_SETUP.md` §7) and notes
  it in the README release status.
- README's EMGFormer table now quotes the measured results of the released
  WL=12000 checkpoints (Avg 14.2/13.9/14.0 deg, improving on the paper's
  14.7/14.2/14.2) with the paper values kept as an inline reference; the
  evaluation recipes pin the training-matched stats pairing so the numbers
  are command-reproducible.
- Fresh-clone reproduction verification: all README evaluation commands and
  all four visualizer modes now run end-to-end on the canonical asset tree
  (center-frame memmap resolution, released-checkpoint config pairing for
  the ResNet-50 fusion, calibration asset location, crop-less-frame
  selection); measured reproduction numbers are recorded in the README
  evaluation notes and docs/code_review_findings_20260820.md.
- Independent structure-review fixes: modality_groups rebuilt from the
  directory layout (9 groups covering all 59 frame fields + 2 episode-level
  entries — the previous 5-group hand copy missed 35 fields); the sentinel
  doc corrected (Incre frame_index is -1, not 0); recording_marks updated
  (Incre carries 8 marks) and the single missing is_last byte at row
  127,369,286 patched (928/928 paired, checksums refreshed); ShowEE beta
  index lossiness documented in episode_fields semantics; vision_index
  declared as a manifest sidecar (regenerable, excluded from checksums);
  validator gains structural checks (groups<->fields<->directories
  consistency, orphan .dat detection, is_first/is_last pairing vs the 928
  beta rows, gesture-label envelope, metadata/manifest schema
  cross-check) plus --no-checksums; both READMEs' verify commands and
  field counts corrected with a copy-paste numpy example; center_frame
  tolerates manifests with implicit filenames.
- Group the unified memmap's flat .dat files into modality subdirectories
  (core/emg/imu/labels/mano/mocap_hands/mocap_wrist/mocap_head/vision) via
  scripts/data/group_memmap_layout.py — pure renames plus manifest pointer
  updates, byte-identical content (evaluation bit-identical; validator all
  green; checksums keys remapped). center_frame resolves episode/split
  arrays through the manifest instead of hardcoded root paths, and the
  validator checksum generation walks subdirectories. Both in-directory
  READMEs rewritten for v3.
- Format-review fixes (independent agent audit): per-file checksums.json
  (64 files) plus scripts/data/validate_memmap.py for one-stop
  schema/episode/source-policy/integrity checks; manifest field_semantics
  extended with timestamp non-uniformity, recording-level is_first/is_last,
  per-source sentinel conventions, and the recording-level episode_fields
  indexing rule; MODALITY_GROUPS mirrored into the manifest; metadata.npz
  normalized (single-encoded string columns, schema_version);
  gesture_classes.json description corrected to 0-59/-1; the dataset share
  README rewritten for v3 with a dependency-free numpy read example; loader
  default emg_field_preference moved to filtered_paper.
- Redesigns the unified memmap schema to v3: positional IMU field names
  (imu_band_left/right, imu_cam_*), mocap_head_valid, dead-field removal,
  a manifest field_semantics block, and an idempotent migration script with
  unit tests; evaluation results are bit-identical across the migration and
  the share was synced (renames + re-upload of missing files after a
  BaiduPCS-Go mv incident that lost five cloud-side files — restored from
  the local source of truth).
- Patches the Baidu NetDisk legacy package in place: the fixed `imu.dat` and
  `manifest.json` were uploaded, downloaded back, verified byte-identical
  (SHA-256), and swapped in; pre-fix files are kept as rollback backups
  under `_imu_fix_20260820/` on the share.
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
