# Asset setup

Data and checkpoints used by the README workflows live under one asset root:

```shell
export EGOEMG_ROOT=/absolute/path/to/egoemg_assets
```

The repository already ships the normalization statistics the configs
reference; copy them into the asset root once:

```shell
mkdir -p "$EGOEMG_ROOT/assets"
cp assets/per_dataset_norm_stats*.json assets/emg_norm_stats.npz \
   "$EGOEMG_ROOT/assets/"
```

Checkpoints install into the repository-local `checkpoints/` directory (set
`EGOEMG_CHECKPOINT_ROOT` to redirect them).

## 1. Data

The data is distributed through the Baidu Netdisk share
<https://pan.baidu.com/s/1aG2e-mHJkmP4KiYtYRcReA> (extraction code `8059`),
rooted at `/EgoEMG_release/`. Download the data folders you need, then place
them into the canonical local layout with one command:

```shell
bash scripts/download/organize_downloads.sh /path/to/downloaded_folders \
  "$EGOEMG_ROOT/data"
```

| Path under `$EGOEMG_ROOT/data/` | Needed for |
| --- | --- |
| `EgoEMG_full_memmap` | EMGFormer training/eval; vision & fusion eval |
| `emg2pose_memmap` | EMG2Pose benchmark |
| `EgoEMG_videos` | `vision` visualization |
| `EgoEMG_crops` | `vision` visualization |

The share additionally carries `dataset_egoemg_zed_videos` (the ShowEE/Incre
ZED recordings); no README workflow needs it.

Baidu caps single files at 128 GB, so the EMG2Pose memmap ships as 100 GB
`.part_XX` chunks; reassemble them after download:

```shell
bash scripts/download/assemble_emg2pose_parts.sh \
  "$EGOEMG_ROOT/data/emg2pose_memmap"
```

Verify a downloaded memmap against the published checksums and schema:

```shell
python scripts/data/validate_memmap.py \
  --memmap-dir "$EGOEMG_ROOT/data/EgoEMG_full_memmap" --checksums
```

The `vision` mode additionally resolves a camera-calibration JSON under
`--data-root` (the preview package ships one; otherwise pass
`--calibration-path` explicitly).

### Preview package

For a visualization smoke test without the full download, a single
self-contained episode is available:

```shell
bash scripts/download/download_egoemg_data.sh \
  --source gdrive "$EGOEMG_ROOT/data/dataset_egoemg_preview"
```

## 2. Checkpoints

```shell
pip install gdown
bash scripts/download/download_checkpoints.sh            # Google Drive
bash scripts/download/download_checkpoints.sh baidupcs   # Baidu Netdisk
```

| README workflow | Checkpoints |
| --- | --- |
| EgoEMG EMGFormer eval (S/M/L) | `egoemg_emgformer_{small,middle,large}.ckpt` |
| EMG2Pose EMGFormer eval (S/M/L) | `emg2pose_emgformer_{small,middle,large}.ckpt` |
| Vision-only eval (ResNet-18 / ViT-S) | `vision_resnet18.ckpt` / `vision_vit_small.ckpt` |
| Fusion eval (ResNet-18 / ViT-S) | `fusion_resnet18_emgfusion_center.ckpt` /
`fusion_vit_emgfusion_center.ckpt` |
| Fusion training init | `vision_resnet18.ckpt` + `egoemg_emgformer_small.ckpt` |

## 3. MANO (visualization only)

The `vision` visualizer projects MANO meshes and needs the separately
licensed MANO model files: pass `--mano-model-path /path/to/mano/models`, or
set `WILOR_PATH` so that `$WILOR_PATH/mano_data/models` exists. A WiLoR
checkpoint is not required. `vision` reads the precomputed crop LMDBs; it
does not crop from bounding boxes at runtime.

## 4. Data integrity

The served `dataset_egoemg_unified` already carries the corrected IMU
channel layout (`[acc, gyro]`; details in `docs/data_known_issues.md` #1).
Verify any copy with the per-file `checksums.json` inside the package
(`sha256sum -c checksums.json`) or the validator command above. Copies saved
before 2026-08-20 can be repaired in place with
`scripts/prepare/fix_egoemg_imu_channel_order.py`.
