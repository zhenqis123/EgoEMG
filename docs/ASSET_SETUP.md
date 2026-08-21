# Legacy asset setup

This repository is a code pre-release, while the **current** EgoEMG release is
still being prepared. The download routes below provide the earlier **legacy
release**. It is the supported asset set for the legacy workflows explicitly
listed in the repository README; it is not a claim that the forthcoming dataset
or every historical experiment is released.

## 1. Choose an asset root

Keep downloaded data outside the Git checkout and export an absolute path. Hydra
creates run directories, so relative paths are not reliable.

```shell
export EMG2POSE_ROOT=/absolute/path/to/egoemg_assets
export EGOEMG_CHECKPOINT_ROOT=/absolute/path/to/emg2pose/checkpoints
```

The second variable is optional: canonical fusion configs default to the
repository-local `checkpoints/` directory created by the checkpoint downloader.

## 2. Preview package

The legacy preview is a single self-contained episode with memmap data,
all-intra video, precomputed crop LMDBs, metadata, and calibration. It is
intended for layout inspection and visualization smoke checks, not training or
paper-level evaluation.

```shell
pip install gdown
bash scripts/download/download_egoemg_data.sh \
  --source gdrive "$EMG2POSE_ROOT/data/dataset_egoemg_preview"

# Or, after logging into baidupcs:
bash scripts/download/download_egoemg_data.sh --source baidupcs
```

Inspect the downloaded package's `manifest.json` and `README.txt` before using
it as a visualizer input. Do not assume that the preview has the complete
release directory layout below.

## 3. Complete legacy data layout

The README's canonical training, evaluation, and full-video visualization
workflows require the complete legacy asset tree. Download it from the legacy
Baidu Netdisk share:

<https://pan.baidu.com/s/1aG2e-mHJkmP4KiYtYRcReA> (extraction code: `8059`).

The top-level `README.txt` in that share maps remote folders to the local layout.
The canonical paths expected by the README are:

```text
$EMG2POSE_ROOT/
├── assets/
│   ├── per_dataset_norm_stats.json
│   ├── per_dataset_norm_stats_unified.json
│   ├── per_dataset_norm_stats_repro_filtered_paper_alias.json
│   └── emg_norm_stats.npz           # pretraining normalization stats
└── data/
    ├── EgoEMG_unified_memmap/       # EMG training/evaluation + labels
    ├── EgoEMG_allintra/             # full-video visualization
    ├── EgoEMG_v2_crops/             # per-episode precomputed crop LMDBs
    ├── reprojection_assets/         # GX010023 camera calibration (viz)
    └── emg_corpus/
        ├── emg2pose_v3/             # EMG2Pose source data
        └── emg2pose_v3_memmap/      # EMG2Pose training/evaluation memmap
```

For Baidu's split EMG2Pose files, reassemble after download:

```shell
bash scripts/download/assemble_emg2pose_parts.sh \
  "$EMG2POSE_ROOT/data/emg_corpus/emg2pose_v3_memmap"
```

## 4. Checkpoints

Download the ten legacy checkpoints into the repository-local `checkpoints/`
directory, or set `EGOEMG_CHECKPOINT_ROOT` to another directory containing the
same filenames.

```shell
pip install gdown
bash scripts/download/download_checkpoints.sh

# Or, after logging into baidupcs:
bash scripts/download/download_checkpoints.sh baidupcs
```

The README's canonical workflows use these files:

| Workflow | Required checkpoint(s) |
| --- | --- |
| EgoEMG EMGFormer evaluation | `egoemg_emgformer_small.ckpt` |
| EMG2Pose EMGFormer evaluation | `emg2pose_emgformer_small.ckpt` |
| ResNet-18 fusion initialization | `vision_resnet18.ckpt`, `egoemg_emgformer_small.ckpt` |
| ResNet-18 fusion evaluation | `fusion_resnet_emgfusion_center.ckpt` |

The legacy bundle also includes middle/large EMGFormer, ViT-S vision, and ViT-S
fusion checkpoints. Historical WiLoR and other research-only configurations are
outside this setup contract.

## 5. MANO and headless visualization

The `vision` visualizer requires separately licensed MANO model files in
addition to the legacy data. Obtain them under their own license and either pass
`--mano-model-path /path/to/mano/models` or set `WILOR_PATH` so that
`$WILOR_PATH/mano_data/models` exists. A WiLoR checkpoint is **not** required
for the canonical ResNet-18 or visualization workflows.

For full videos, use the complete layout above. `vision` reads the existing
precomputed LMDB crops; it does not derive crops from bounding boxes at runtime.

## 6. Verify before a long run

- Run `python scripts/release/audit_portability.py` to identify research-only
  configs; canonical README configs should not require developer `logs/` or
  `test_results/` paths.
- Run `python scripts/viz/visualize_dataset.py vision --help` to check Python
  dependencies and CLI availability.
- Start with the preview package for a visualization smoke check, then use the
  complete legacy tree for training and benchmark evaluation.


## 7. IMU channel-order fix (2026-08-20)

The EgoEMG source wrist band streams its IMU gyro-first
(`[gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]`, gyro_x dead/always 0), but
the unified memmap documents — and all consumers expect — accel-first
`[acc, gyro]`. Legacy-release copies of `dataset_egoemg_unified/` uploaded
before 2026-08-20 store the EgoEMG rows (episodes 0–40) with the halves
swapped; the ShowEE/Incre rows were always correct.

**Check whether a copy needs the fix** (gravity should sit in the first three
channels; pre-fix copies read ~0.2 there):

```shell
python - <<'PY'
import numpy as np, json
root = "<path to EgoEMG_unified_memmap>"
m = json.load(open(f"{root}/manifest.json"))
n = m["fields"]["imu"]["shape"][0]
imu = np.memmap(f"{root}/imu.dat", dtype="float32", mode="r", shape=(n, 6))
p50 = np.percentile(np.linalg.norm(imu[:1_000_000:10, :3], axis=1), 50)
print("first-half |.| p50 =", round(float(p50), 2), "->", "FIXED" if p50 > 7 else "NEEDS FIX")
PY
```

**Repair a downloaded copy in place** (reversible; keeps a 3.2 GB backup next
to the file):

```shell
python scripts/prepare/fix_egoemg_imu_channel_order.py \
  --memmap-dir <path to EgoEMG_unified_memmap> --apply
```

**Verify against the canonical fixed files** (v3 schema names; the content
of `imu_band_left.dat` is byte-identical to the pre-v3 `imu.dat`):

| File | SHA-256 |
| --- | --- |
| `imu_band_left.dat` | `9a0bb4565272f746f35b6a2922e7ba421e4f485fb87f9f2c01260df825b01a6e` |

The full per-file checksum table lives in the share itself as
`dataset_egoemg_unified/checksums.json` (64 files); verify any download
with `sha256sum -c checksums.json` inside the directory, or run
`python scripts/data/validate_memmap.py --memmap-dir <dir> --checksums`
for schema/episode/source-policy checks as well.

**Directory layout (local, grouped).** The working directory groups the 61
`.dat` files into modality subdirectories
(`core/ emg/ imu/ labels/ mano/ mocap_hands/ mocap_wrist/ mocap_head/ vision/`;
`scripts/data/group_memmap_layout.py` performs the move and rewrites the
manifest). The share currently still carries the flat v3 layout — the two
are byte-identical in content and interchangeable via the manifest's
`filename` pointers; the release repackage will adopt the grouped layout.

**Schema v3 (2026-08-20).** The share serves the v3 schema
(`format_version: egoemg_v3_memmap`): IMU fields renamed to the positional
taxonomy (`imu_band_left/right`, `imu_cam_head`, `imu_cam_wrist_left/right`),
`mocap_head_tracked` → `mocap_head_valid`, dead fields dropped
(`task_index`, `source_index`, `is_terminal`, `emg_right_filtered`,
`timestamp`), and a comprehensive `field_semantics` block added to the
manifest. Older v2 copies can be upgraded in place with
`python scripts/data/migrate_unified_memmap_v3.py --memmap-dir <dir> --apply`.

**Maintainer: Baidu NetDisk copy — patched 2026-08-20.** The cloud package
(`/EgoEMG_release/dataset_egoemg_unified/`) now serves the fixed `imu.dat`
and `manifest.json`. Both were uploaded to a staging dir, downloaded back,
and verified byte-identical (SHA-256 above) before the swap; the pre-fix
files remain as rollback backups in
`/EgoEMG_release/dataset_egoemg_unified/_imu_fix_20260820/*.pre-fix.bak`
(delete once the release is confirmed stable). To roll back, `mv` the
`.pre-fix.bak` files back over the main-dir names.

Fresh downloads therefore need no repair; only copies saved before
2026-08-20 need the in-place fix command above.

**Packaging note (~44 GB of sparse zeros).** The Incre source rows leave
many fields entirely zero; the directory stores them as sparse holes on
the build machine (`du` 182 GB vs 226 GB apparent). Archives that do not
preserve sparse files will materialize those zeros. When re-packaging use
`tar -S` (or `rsync -S`), or split archives per source so the Incre volume
can document its zero fields explicitly.

Post-fix verification evidence (original parquet bitwise comparison,
per-episode gravity statistics, ShowEE/Incre guard checksums) is recorded in
`docs/data_known_issues.md` #1 and
`scripts/release/imu_verify_report_windows_original.json`.
