# EgoEMG WiLoR Training Guide

This guide documents the EgoEMG vision dataset path used to fine-tune WiLoR.
It covers dataset preparation, visualization checks, training, and the
implementation constraints that keep startup and decoding fast.

## Data Layout

The training entrypoint expects three data roots:

- `data_location`: EgoEMG memmap directory, for example `data/EgoEMG_unified_memmap`.
- `video_root`: raw EgoEMG dataset root containing the original head-view paths and
  reprojection assets, for example `data/EgoEMG`.
- `allintra_root`: all-intra re-encoded head-view videos, for example
  `data/EgoEMG_allintra`.

`EgoEmgVisionDataset` reads labels and transforms from the memmap, but reads
head-view frames from all-intra videos only. Video decoding is done with `decord`
through `egoemg.video_io.DecordVideoReaderCache`. Missing all-intra videos are
treated as errors; there is intentionally no OpenCV or original-video fallback
in this path.

## One-Time Preparation

Build the sidecar vision index once. This avoids scanning every frame at dataset
startup.

```bash
python scripts/data/build_egoemg_vision_index.py \
    --memmap-dir data/EgoEMG_unified_memmap \
    --output-dir data/EgoEMG_unified_memmap/vision_index
```

The index stores valid frame ids per split, episode, and hand. At runtime the
dataset selects samples from this sidecar according to `allowed_*_splits`,
`target_hand`, `stride`, and `index_limit`.

If you need to create all-intra videos, use the repository conversion script:

```bash
python scripts/prepare/reencode_egoemg_webcam_allintra.py \
    --memmap-dir data/EgoEMG_unified_memmap \
    --data-root data/EgoEMG \
    --output-root data/EgoEMG_allintra
```

## Dataset Smoke Test

Use the visualization script before launching training. It renders the exact
samples emitted by `EgoEmgVisionDataset`, including the raw-frame supervision,
training crop, bbox, and normalized patch keypoints.

```bash
python scripts/viz/visualize_egoemg_vision_dataset.py \
    --memmap-dir data/EgoEMG_unified_memmap \
    --video-root data/EgoEMG \
    --allintra-root data/EgoEMG_allintra \
    --vision-index-dir data/EgoEMG_unified_memmap/vision_index \
    --output-dir /tmp/egoemg_vision_dataset_viz \
    --num-samples 16 \
    --target-hand both
```

For quick startup profiling, reduce to one or two samples:

```bash
python scripts/viz/visualize_egoemg_vision_dataset.py \
    --memmap-dir data/EgoEMG_unified_memmap \
    --video-root data/EgoEMG \
    --allintra-root data/EgoEMG_allintra \
    --vision-index-dir data/EgoEMG_unified_memmap/vision_index \
    --output-dir /tmp/egoemg_vision_dataset_viz \
    --num-samples 1 \
    --target-hand both
```

Expected startup behavior with an installed sidecar index is fast. The logged
`build_index` stage should be milliseconds, not minutes.

## Dataset API

Direct usage:

```python
from pathlib import Path

from egoemg.datasets.egoemg_vision_dataset import EgoEmgVisionDataset

dataset = EgoEmgVisionDataset(
    memmap_dir=Path("data/EgoEMG_unified_memmap"),
    video_root=Path("data/EgoEMG"),
    allintra_root=Path("data/EgoEMG_allintra"),
    vision_index_dir=Path("data/EgoEMG_unified_memmap/vision_index"),
    target_hand="both",
    allowed_splits=["train"],
    stride=30,
)

sample = dataset[0]
```

Important sample fields:

- `img`: normalized RGB training patch in WiLoR format.
- `keypoints_2d`: normalized 2D MANO/OpenPose hand keypoints in patch space.
- `keypoints_3d`: camera-space 3D keypoints.
- `mano_params`, `global_orient`, `hand_pose`, `betas`: MANO supervision.
- `orig_keypoints_2d`, `orig_markers_2d`, `bbox`: raw-frame debug values.
- `frame_bgr`: optional raw frame, returned only when `return_frame_bgr=True`.

`index_limit` is intended for smoke tests and visualization. Do not use it for
full training.

## MANO Semantics

EgoEMG generated MANO labels use a single canonical MANO-right semantic for both
hands.

- Decode `generated_mano_right_pose` with MANO-right semantics.
- Decode `generated_mano_left_pose` with MANO-right semantics as well.
- Recover the displayed left hand by applying the left-hand reflection at
  visualization/alignment time.

Do not interpret EgoEMG left-hand MANO pose labels as MANO-left pose semantics.

`EgoEmgVisionDataset` does not initialize or run MANO. It emits raw MANO pose
and shape labels plus projected mocap keypoint supervision. MANO initialization
and MANO forward passes stay inside the WiLoR model path.

The default `mano_model_path` points to the existing MANO asset directory:

```yaml
mano_model_path: ../WiLoR/mano_data
```

This path is used by the WiLoR model for MANO assets and can be overridden in
config or CLI.

## Training WiLoR

The training entrypoint is:

```bash
python -m egoemg.train \
    experiment=fusion/vision_resnet_middle_egoemg_showee \
    data_location=data/EgoEMG_unified_memmap \
    video_root=data/EgoEMG \
    allintra_root=data/EgoEMG_allintra \
    vision_index_dir=data/EgoEMG_unified_memmap/vision_index \
    mano_model_path=../WiLoR/mano_data \
    wilor_checkpoint_path=../WiLoR/pretrained_models/wilor_final.ckpt \
    train=True \
    eval=True
```

The defaults live in `config/experiment/fusion/vision_resnet_middle_egoemg_showee.yaml`. Common overrides:

- `devices=[0]`: GPU selection.
- `batch_size=64 val_batch_size=64`: train and eval batch sizes.
- `num_workers=8`: dataloader workers.
- `stride=30 val_stride=300 test_stride=300`: sampling density per split.
- `target_hand=both`: train both hands, or restrict to `left` / `right`.
- `checkpoint=/path/to/ckpt`: evaluate a trained checkpoint.
- `train=False eval=True`: evaluation-only mode.

Example evaluation-only run:

```bash
python -m egoemg.train \
    experiment=fusion/vision_resnet_middle_egoemg_showee \
    data_location=data/EgoEMG_unified_memmap \
    video_root=data/EgoEMG \
    allintra_root=data/EgoEMG_allintra \
    vision_index_dir=data/EgoEMG_unified_memmap/vision_index \
    checkpoint=/path/to/checkpoints/last.ckpt \
    train=False \
    eval=True
```

## Training Flow

`python -m egoemg.train` (with a fusion or vision-only experiment config)
performs the following steps:

1. Loads `config/experiment/fusion/vision_resnet_middle_egoemg_showee.yaml` through Hydra.
2. Builds the WiLoR config from `WiLoR/pretrained_models/model_config.yaml`.
3. Creates train/val/test `EgoEmgVisionDataset` instances.
4. Loads all-intra video frames with `decord`.
5. Emits WiLoR-native batches:
   `{"img": batch, "mocap": {"hand_pose", "betas", "global_orient"}}`.
6. Initializes `EgoEMGWiLoRModule`.
7. Optionally loads `wilor_checkpoint_path`.
8. Trains with Lightning and checkpoints on `val/loss`.
9. Loads the best checkpoint and runs test evaluation when `eval=True`.

Hydra writes outputs under the run directory. Checkpoints are saved under
`checkpoints/`, and TensorBoard logs are saved under `tensorboard/`.

## Performance Notes

- Build `vision_index` once. Runtime dataset construction should not scan all
  frames.
- Use all-intra videos only. Random frame access on original long-GOP videos is
  slow and intentionally unsupported in this path.
- Keep `decord` installed in the active environment.
- The first `getitem` in a process may include MANO asset loading and layer
  buffer initialization. Subsequent samples should be much faster.
- Increase `num_workers` only after verifying storage bandwidth; each worker has
  its own process-local video reader cache.

## Related Scripts

- `scripts/data/build_egoemg_vision_index.py`: one-time sidecar index generation.
- `scripts/prepare/reencode_egoemg_webcam_allintra.py`: all-intra webcam conversion.
- `scripts/viz/visualize_egoemg_vision_dataset.py`: dataset sample debug.
- `scripts/visualize_egoemg_mesh.py`: world-space MANO mesh projection debug.
- `egoemg/train.py`: the unified training entrypoint; vision/fusion training
  is launched with `experiment=fusion/...` or a vision-only experiment config.

## Mixed EgoEMG and ShowEE training

The converted ShowEE shard is exposed through
`dataset=egoemg_showee_angle_regression`. It contributes independent left- and
right-hand samples to train, validation, and test. Its 20 finger angles use the
same canonical MANO-right semantics as EgoEMG. ShowEE does not provide reliable
wrist pitch/yaw, so channels 20 and 21 are zero-filled and masked from losses
and metrics only for samples whose `dataset_name` is `showee`.

Ready-to-run experiment deltas are provided for all three main modes:

```bash
# EMG-only
python -m egoemg.train experiment=emgformer/regression_egoemg_showee

# Vision-only
python -m egoemg.train \
    experiment=fusion/vision_resnet_middle_egoemg_showee

# EMG + vision fusion
python -m egoemg.train \
    experiment=fusion/fusion_rn18_s_center_8ch_egoemg_showee
```

ShowEE vision training reads the precomputed crops from
`data/ShowEE_202607_crops`; the all-intra videos remain available
for rebuilding or checking those crops. Dataset-specific filtered-paper EMG
statistics are stored in
`assets/per_dataset_norm_stats_repro_filtered_paper_alias.json`.
