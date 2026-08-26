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

> **For the EMG-to-pose experiment you only need the two memmaps.** Download
> `EgoEMG_full_memmap` (+ `emg2pose_memmap` for the EMG2Pose benchmark). The
> `EgoEMG_videos` / `EgoEMG_crops` folders are used **only** by the `vision`
> visualization mode — skip them for EMG training/eval. Doing so keeps the
> download much smaller and avoids the share's per-user transfer quota.
>
> The data is distributed through the **Baidu Netdisk** share (public, no
> per-user download quota). Google Drive is a **legacy mirror** that only
> carries the one-episode preview package and the checkpoints — those are an
> **older version** and will be refreshed. Use the Baidu link above.

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

For a visualization / eval / smoke-train test without the full download, a
small **preview shard** (3 episodes + vision, flat v3 layout) is published at
`/EgoEMG_release/dataset_egoemg_preview` on the Baidu share. Download that
folder from the share <https://pan.baidu.com/s/1aG2e-mHJkmP4KiYtYRcReA>
(extraction code `8059`) and place it at
`$EGOEMG_ROOT/data/dataset_egoemg_preview`.

The package keeps the same layout as the full dataset so the README commands
drop in directly (point `--memmap-dir` at
`$EGOEMG_ROOT/data/dataset_egoemg_preview/data/memmap_data`):

```text
data/
  memmap_data/                    flat v3 memmap (manifest.json + *.dat + metadata.npz)
  webcam_videos/                  episode_XXXXXX_head_allintra.mp4
  pre-crop_webcam_videoframes/    episode_XXXXXX.lmdb + manifest.json (patch_size 256)
  GX010023_standard_calibration.json
```

It can also be **built locally** by slicing whole episodes out of the full
unified memmap (needs the same source the builder reads, i.e. not applicable
to a fresh download):

```shell
python scripts/data/build_egoemg_preview_memmap.py \
  --memmap-dir /path/to/EgoEMG_full_memmap \
  --out /path/to/dataset_egoemg_preview \
  --allintra-root /path/to/EgoEMG_videos \
  --crops-root /path/to/EgoEMG_crops \
  --copy-videos --copy-crops          # omit to symlink instead
```

Because the shard carries a single source and a subset of splits, validate it
with the relaxed check:

```shell
python scripts/data/validate_memmap.py \
  --memmap-dir /path/to/dataset_egoemg_preview/data/memmap_data \
  --allow-partial-sources
```

## 2. Checkpoints

Download the `checkpoints` folder under `/EgoEMG_release/` from the Baidu share
<https://pan.baidu.com/s/1aG2e-mHJkmP4KiYtYRcReA> (extraction code `8059`) and
place it in the repository-local `checkpoints/` directory (set
`EGOEMG_CHECKPOINT_ROOT` to redirect it).

| README workflow | EMG channels | Checkpoints |
| --- | --- | --- |
| EgoEMG EMGFormer eval (S/M/L) | 8 | `egoemg_emgformer_{small,middle,large}.ckpt` |
| EMG2Pose EMGFormer eval (S/M/L) | 16 | `emg2pose_emgformer_{small,middle,large}.ckpt` |
| Vision-only eval (ResNet-18 / ViT-S) | — | `vision_resnet18.ckpt` / `vision_vit_small.ckpt` |
| Fusion eval (ResNet-18 / ViT-S) | 16 | `fusion_resnet18_emgfusion_center.ckpt` /
`fusion_vit_emgfusion_center.ckpt` |
| Fusion training init | — | `vision_resnet18.ckpt` + `egoemg_emgformer_small.ckpt` |

> **Channels = the EMG input channel count, which is NOT uniform across
> checkpoint families.** The EgoEMG EMGFormer checkpoints are **8-channel**
> (single-hand `target_hand` layout); the EMG2Pose and fusion checkpoints are
> **16-channel** (bilateral `emg2pose_interpolate16` layout). A config and its
> checkpoint must come from the **same family** — pairing a 16-channel
> checkpoint with an 8-channel config (or vice-versa) fails with a featurizer
> shape mismatch. `egoemg_*` and `emg2pose_*` checkpoints are not
> interchangeable even though both use `_small`/`_middle`/`_large` names.

## 3. MANO (visualization only)

The `vision` visualizer projects MANO meshes and needs the separately
licensed MANO model files: pass `--mano-model-path /path/to/mano/models`, or
set `WILOR_PATH` so that `$WILOR_PATH/mano_data/models` exists. A WiLoR
checkpoint is not required. `vision` reads the precomputed crop LMDBs; it
does not crop from bounding boxes at runtime.

## 4. Data integrity

The served `dataset_egoemg_unified` already carries the corrected IMU
channel layout (`[acc, gyro]`). Verify any copy with the per-file
`checksums.json` inside the package
(`sha256sum -c checksums.json`) or the validator command above. Copies saved
before 2026-08-20 can be repaired in place with
`scripts/prepare/fix_egoemg_imu_channel_order.py`.
