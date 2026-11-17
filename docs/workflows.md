# EgoEMG training and evaluation

Prepare assets, train models and evaluate checkpoints for EgoEMG and EMG2Pose.
Start with the [repository setup](../README.md#setup), then choose a workflow below.

[Prepare assets](#prepare-assets) · [Train](#train) · [Evaluate](#evaluate) ·
[Model variants](#model-variants)

## Prepare assets

To check the workflow on two short clips, start with the
[demo subset](dataset/README.md#demo-subset). It includes commands for a brief
training run and pretrained EMG, vision and fusion inference.

Install from the source checkout as shown in the repository README. Use the official
session dataset, its bundled evaluation protocol, and training normalization
statistics. `EGOEMG_ROOT` contains `data/EgoEMG_dataset/` and `assets/`;
`EGOEMG_CHECKPOINT_ROOT` points to `pretrained_models/egoemg/`. See [Setup](../README.md#setup)
for download links and extraction codes.

### Asset layout

```text
$EGOEMG_ROOT/
  assets/                         training normalization statistics
  data/EgoEMG_dataset/
    dataset.json
    sessions/<session_id>/session.json
    assets/                       bundled dataset metadata and statistics
    fullframe_256_index/           sparse evaluation cache
    fullframe_256_30fps/           dense training and temporal cache
  pretrained_models/egoemg/        released EgoEMG checkpoints
  pretrained_models/emg2pose/      released EMG2Pose EMGFormer S/M/L
  pretrained_models/dinov2/        optional local DINOv2 initialization
  pretrained_models/markers2mano/   marker-to-MANO conversion
```

Session manifests define the authoritative relative paths. Keep those directory
relationships and copy only the two training normalization JSONs to `$EGOEMG_ROOT/assets/`
as shown in the repository README. Reuse training statistics for val and test.

EMG2Pose uses a [separate corpus](#emg2pose-corpus).

### Vision caches

EMG-only runs do not require image caches. Build vision caches at their final
location: signatures include source paths and file metadata. Use FFmpeg
all-intra proxies and decord.

```bash
python -m scripts.data.session.build_sparse_fullframe_cache \
  --dataset-dir "$EGOEMG_ROOT/data/EgoEMG_dataset"
python -m scripts.data.session.build_dense_fullframe_cache \
  --dataset-dir "$EGOEMG_ROOT/data/EgoEMG_dataset" \
  --output-dir "$EGOEMG_ROOT/data/EgoEMG_dataset/fullframe_256_30fps"
```

Vision/fusion evaluation uses the sparse cache. Dense30 training and
temporal evaluation additionally require the dense cache. Override training
with `dense_frame_cache=/path/to/cache`, or evaluation with
`--dense-cache /path/to/cache`.

### EMG2Pose corpus

[Download the corpus](https://pan.baidu.com/s/1iBAwFsbgH8wcqvNcVrdTfg?pwd=eg8m) from Baidu Netdisk (code: `eg8m`).

Place the EMG2Pose corpus under
`$EGOEMG_ROOT/data/emg_corpus/emg2pose_v3_memmap/`. This separate corpus is
needed only for EMG2Pose training and evaluation.

The download contains five `*.part_XX` files for the two largest arrays.
Restore them with the included assembler:

```bash
python "$EGOEMG_ROOT/data/emg_corpus/emg2pose_v3_memmap/assemble_parts.py"
```

Keep `manifest.json`, `metadata.npz`, `metadata_summary.json` and
`per_dataset_norm_stats.json` beside the restored `.dat` files. The assembler
checks part numbering and declared sizes. Allow space for both the downloaded
parts and the restored arrays.

### Released checkpoint bundle

[Download the checkpoints](https://pan.baidu.com/s/1DMgtNnJdeTZNh1V975IBBQ?pwd=eg8m) from Baidu Netdisk (code: `eg8m`).
Place the downloaded `pretrained_models/` folder directly under `$EGOEMG_ROOT/`.

Keep weights outside Git. Each fusion model needs its standalone vision checkpoint
and `emgformer_s.ckpt` in the same directory for architecture restoration;
evaluation strictly loads its full fusion state.

The bundle contains nine EgoEMG, three EMG2Pose and one Markers2MANO checkpoints:

| Directory under `pretrained_models/` | Checkpoints |
|---|---|
| `egoemg/` | `emgformer_s.ckpt` |
| `egoemg/` | `vision_vit_s.ckpt`, `vision_vit_b.ckpt`, `vision_vit_l.ckpt` |
| `egoemg/` | `fusion_vit_s_emgformer_s.ckpt`, `fusion_vit_b_emgformer_s.ckpt`, `fusion_vit_l_emgformer_s.ckpt` |
| `egoemg/` | `vision_vit_s_temporal_17f.ckpt`, `fusion_vit_s_temporal_17f_emgformer_s.ckpt` |
| `emg2pose/` | `emgformer_small.ckpt`, `emgformer_middle.ckpt`, `emgformer_large.ckpt` |
| `markers2mano/` | `markers2mano.ckpt` |

The EgoEMG EMGFormer-S checkpoint uses filtered EMG and augmentation.

## Train

Train on the training split, use validation for checkpoint selection and early
stopping, and evaluate the test split after selection. EgoEMG recipes
default to one GPU; the EMG2Pose stride390 recipe retains its recorded six-GPU
training budget. Adjust hardware settings together with gradient accumulation
and validation intervals to retain the intended optimizer-step schedule.
Append `--cfg job` to inspect a composed recipe without starting training.

```bash
python -m egoemg.train experiment=emgformer/emgformer_s
python -m egoemg.train experiment=fusion/vision_vit_s
python -m egoemg.train experiment=fusion/fusion_vit_s \
  fusion_vision_checkpoint=/path/to/selected_vision.ckpt \
  fusion_emg_checkpoint=/path/to/selected_emg.ckpt
```

The EgoEMG EMGFormer recipe uses the native hand timeline and 22 angle
targets. Finger and wrist validity are applied separately; stored wrist
degrees are converted to radians. Windows respect recordings, split boundaries and discontinuous clocks.

The ViT-S recipe starts from DINOv2. For offline initialization, set
`+module.pretrained_path="$EGOEMG_ROOT/pretrained_models/dinov2/dinov2_vits14.safetensors"`.
Train the vision model with dense30 sampling before using its checkpoint to
initialize fusion.

Residual fusion initializes both the vision and EMG branches. It uses
vision features, temporal EMG pooling, fusion projections, hand ID, and
separate residual output heads for the left and right hands. The training
objective is MAE on the predicted angles. Initialize fusion from
validation-selected standalone checkpoints.

## Evaluate

### Evaluate EgoEMG checkpoints

Use the dedicated evaluator to load selected checkpoints and compute metrics.

```bash
python -m egoemg.evaluate \
  --checkpoint-dir "$EGOEMG_CHECKPOINT_ROOT" \
  --dataset-dir "$EGOEMG_ROOT/data/EgoEMG_dataset" \
  --models emgformer_s vision_vit_s fusion_vit_s_emgformer_s \
  --split test --device cuda:0 --workers 4 \
  --output-dir logs/evaluation/egoemg
```

For a loading/inference smoke check add `--max-batches 1 --batch-size 1 --workers 0`
and use a separate output directory. Such partial outputs are not benchmark
results. Temporal checkpoints use `<dataset-dir>/fullframe_256_30fps` by default;
set `--dense-cache` to use another cache location.
See [Released checkpoint bundle](#released-checkpoint-bundle) for required companion weights.

Output MAE is in radians and weighted by valid-angle counts. Multiply by
`180 / pi` for degrees. Finger, wrist, split, hand, and subject summaries are
reported separately. Each model writes the following under `<output-dir>/<model>/test/`:

| File | Contents |
|---|---|
| `results.csv` | Aggregate metrics by split and hand |
| `joint_results.csv` | Per-angle MAE and valid-angle counts |
| `group_results.csv` | Per-subject and per-session metrics |

### Evaluate EMG2Pose checkpoints

Use `egoemg.test_analysis` with the EMG2Pose corpus. The `egoemg.evaluate`
command above reads EgoEMG session data and is a separate workflow.

```bash
python -m egoemg.test_analysis \
  experiment=emgformer/emg2pose_emgformer_s_stride390 \
  checkpoint="$EGOEMG_ROOT/pretrained_models/emg2pose/emgformer_small.ckpt" \
  per_user=true \
  hydra.run.dir=logs/evaluation/emg2pose_s
```

Choose a matching recipe and checkpoint:

| Size | Experiment | Checkpoint under `pretrained_models/emg2pose/` |
|---|---|---|
| S | `emgformer/emg2pose_emgformer_s_stride390` | `emgformer_small.ckpt` |
| M | `emgformer/emg2pose_emgformer_m_stride390` | `emgformer_middle.ckpt` |
| L | `emgformer/emg2pose_emgformer_l_stride390` | `emgformer_large.ckpt` |

Keep the assembled corpus and its normalization statistics at the locations
specified in [EMG2Pose corpus](#emg2pose-corpus). Evaluation uses one device
for inference.
`per_user=true` computes a mean and population SD across users within each
of the User, Stage and User+Stage test groups. These are the
`per_user_mae_mean` and `per_user_mae_std` columns in `results.csv`, in radians.
Multiply both by `180 / pi` for degrees. Results go to the directory selected
by `hydra.run.dir`.

## Model variants

| Experiment | Purpose |
|---|---|
| `emgformer/emgformer_s` | EMGFormer-S; M/L via decoder overrides |
| `emgformer/emg2pose_emgformer_s_stride390` | EMG2Pose EMGFormer-S; stride 390 and fingertip loss 0.01 |
| `emgformer/emg2pose_emgformer_m_stride390` | EMG2Pose EMGFormer-M; stride 390 and fingertip loss 0.01 |
| `emgformer/emg2pose_emgformer_l_stride390` | EMG2Pose EMGFormer-L; stride 390 and fingertip loss 0.01 |
| `fusion/vision_vit_s` | Single-frame ViT-S vision baseline |
| `fusion/fusion_vit_s` | Residual EMG + vision fusion |
| `fusion/direct_fusion_vit_s` | Direct-fusion ablation |
| `fusion/temporal_vision_vit_s` | 17-frame ViT-S temporal vision |
| `fusion/temporal_fusion_vit_s` | 17-frame temporal vision + EMG residual fusion |

The following sections explain model-size overrides, corpus inputs and
fusion initialization.

EMGFormer-M/L reuse the same recipe with a decoder override:

```bash
python -m egoemg.train experiment=emgformer/emgformer_s \
  module/decoder=transformer/preset/middle_aggressive
python -m egoemg.train experiment=emgformer/emgformer_s \
  module/decoder=transformer/preset/large_aggressive
```

The EMG2Pose release provides separate S/M/L entrypoints with 16 EMG channels
and 20 finger-angle outputs. The L entrypoint reproduces the recorded
stride390/FK0.01 settings; S and M reuse the same data and training budget,
changing only the decoder size:
7,790-sample windows, training stride 390, non-overlapping validation/test
windows, train-derived per-dataset normalization, noncausal attention, and the
configured signal augmentations. The training objective is angular MAE plus a
fingertip-distance term weighted by 0.01. Checkpoints are selected using
`val_user_macro_mae` (valid-angle MAE per user, then equal-weight users).

```bash
python -m egoemg.train experiment=emgformer/emg2pose_emgformer_s_stride390
python -m egoemg.train experiment=emgformer/emg2pose_emgformer_m_stride390
python -m egoemg.train experiment=emgformer/emg2pose_emgformer_l_stride390
```

Provide the official recording-level corpus under
`$EGOEMG_ROOT/data/emg_corpus/emg2pose_v3_memmap/` and its training statistics at
`$EGOEMG_ROOT/data/emg_corpus/emg2pose_v3_memmap/per_dataset_norm_stats.json`. The recipe retains six GPUs,
batch size 250 per GPU, and two-batch gradient accumulation: an effective batch
of 3,000 per optimizer update. Training lasts 33,600 optimizer steps; validation
runs every 224 optimizer steps (448 microbatches). The step-based cosine schedule
uses 224 steps per period for 150 periods, decaying from 5e-4 to 5e-5 without
warmup. Test evaluation is disabled during training.

For one GPU with the same effective batch and validation cadence, append
these overrides to the training command:

```text
trainer.devices=[0] trainer.strategy=auto trainer.accumulate_grad_batches=12 \
  trainer.val_check_interval=2688
```

For ViT-B/L, override `module.backbone_type=vit_base` or `vit_large` in the
single-frame vision recipe. A ResNet uses the same sampling and supervision:

```bash
python -m egoemg.train experiment=fusion/vision_vit_s \
  module._target_=egoemg.models.modules.bimanual_vision.BimanualVisionResNetPose \
  module.backbone_type=resnet50
```

Use `resnet18` or `resnet152` for the other reported ResNets. Fusion restores its
vision architecture from `fusion_vision_checkpoint`; pass the matching
validation-selected standalone vision branch and EMG branch. Direct fusion uses
the same data, initialization paths, and optimization settings as residual
fusion, changing only the prediction module.

The temporal recipes use 17 RGB frames at 250 ms intervals over four seconds.
Temporal vision starts from DINOv2 and trains the visual and temporal modules
jointly. Offline DINOv2 weights can be supplied with
`+module.vision_config.pretrained_path="$EGOEMG_ROOT/pretrained_models/dinov2/dinov2_vits14.safetensors"`.
Temporal fusion loads a validation-selected temporal vision checkpoint:

```bash
python -m egoemg.train experiment=fusion/temporal_fusion_vit_s \
  fusion_vision_checkpoint=/path/to/selected_temporal_vision.ckpt \
  fusion_emg_checkpoint=/path/to/selected_emg.ckpt
```

These recipes start new training runs. Released checkpoints retain their
original training and selection history.

## Configuration changes

Run commands from the repository root. Public experiments inherit shared
defaults from `config/lineage/` and select their model and dataset under
`config/experiment/`. Use the [recipe list](#model-variants) to choose an
entrypoint, then append Hydra overrides as shown above. Add `--cfg job` to
inspect the configuration before a run.

Write outputs under `logs/` and keep checkpoints under
`pretrained_models/<model-family>/`.
