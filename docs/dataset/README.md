# EgoEMG dataset

The official EgoEMG dataset contains 81 sessions from 68 participants and
160 recorded hand streams. `dataset.json` is the entry point. All paths in
its session list are relative to this directory; array paths in a session
manifest are relative to that session directory.

**Dataset version: V1.** [Download from Baidu Netdisk](https://pan.baidu.com/s/1xBd7Ghwg0Uq9mce5mHRwKA?pwd=eg8m)
(extraction code: `eg8m`; no scheduled expiry).

## Demo subset

[Download the demo](https://pan.baidu.com/s/1mzgVqJUaUrMezByxKI0Ktw?pwd=eg8m) from Baidu Netdisk (code: `eg8m`).
The `EgoEMG_demo.tar.gz` archive is **10.35 GB**; extraction requires about
**13.1 GB**, with additional space for locally generated image caches.

The demo contains two continuous 10-minute clips:

| Activity | Source session | Duration |
|---|---|---|
| Bare-hand gestures | `sess_20260404_153935` | 10 min |
| Hand–object interaction | `sess_20261016_161026` | 10 min |

The clips retain both hands, available camera streams, hand crops, annotations
and calibration. Video is trimmed at frame boundaries and re-encoded
at its original resolution. Use this subset for functional checks and the full
dataset for benchmark results.

Install the project environment, then extract the archive and run these commands
from the code repository root:

```bash
export EGOEMG_ROOT=/path/to/egoemg_assets
export EGOEMG_CHECKPOINT_ROOT="$EGOEMG_ROOT/pretrained_models/egoemg"
mkdir -p "$EGOEMG_ROOT/data" "$EGOEMG_ROOT/assets"
tar -xzf /path/to/EgoEMG_demo.tar.gz -C "$EGOEMG_ROOT/data"
export EGOEMG_DEMO="$EGOEMG_ROOT/data/EgoEMG_demo"
cp "$EGOEMG_DEMO"/assets/egoemg_emg_normalization_{filtered,raw}.json "$EGOEMG_ROOT/assets/"
python "$EGOEMG_DEMO/readers/inspect_dataset.py" \
  --root "$EGOEMG_DEMO" --all-sessions --media
python scripts/tools/dataset_viewer/launch.py \
  --dataset-dir "$EGOEMG_DEMO" --port 8793
```

Open `http://127.0.0.1:8793/` and select either clip.

To check EMGFormer training and validation with two batches each:

```bash
python -m egoemg.train experiment=emgformer/emgformer_s \
  egoemg_session_v2_dir="$EGOEMG_DEMO" \
  evaluation_partition_path="$EGOEMG_DEMO/assets/egoemg_evaluation_protocol.json" \
  batch_size=2 val_batch_size=2 num_workers=0 \
  +trainer.fast_dev_run=2 'trainer.devices=[0]' \
  hydra.run.dir=logs/demo/train
```

For pretrained EMG, vision and fusion inference, download the three S checkpoints
(`emgformer_s.ckpt`, `vision_vit_s.ckpt`, `fusion_vit_s_emgformer_s.ckpt`),
then prepare the sparse image cache and run a short evaluation check:

```bash
python -m scripts.data.session.build_sparse_fullframe_cache \
  --dataset-dir "$EGOEMG_DEMO" --workers 2
python -m egoemg.evaluate \
  --checkpoint-dir "$EGOEMG_CHECKPOINT_ROOT" \
  --dataset-dir "$EGOEMG_DEMO" \
  --models emgformer_s vision_vit_s fusion_vit_s_emgformer_s \
  --split test --device cuda:0 --workers 0 --batch-size 1 --max-batches 1 \
  --output-dir logs/demo/evaluation
```

The reader reports both sessions, the training run completes its train and
validation batches, and evaluation writes CSV files for each model.

## Releases

V1 is the version described here. V2 and V3 describe subsequent additions;
check the published download announcement for availability.

| Version | Contents |
|---|---|
| V1 | EMG, available IMU, RGB and hand crops, hand poses, object IDs, meshes and textures, available SAM3 masks, all camera geometry |
| V2 | V1 plus available depth maps |
| V3 | V2 plus object poses |

The same public download directory is updated in place. Each published version
contains everything needed for that version. V1 includes camera intrinsics, extrinsics, distortion parameters and
available camera-world poses. Depth is added in V2, object poses in V3.

## Scope and access

The manuscript describes 68 participants, 25.9 hours, 60 bare-hand gesture
classes, and hand-object interactions. Recorded modalities include bilateral
surface EMG, IMU, RGB, available depth and motion-capture-derived hand labels.
The version table above defines the contents of each version.

The manuscript reports institutional ethics approval and informed consent for
collection and intended research use. V1 is publicly available through the
links above and includes `RELEASE_MANIFEST.json`. EgoEMG-authored recordings
and annotations are licensed under [CC BY-NC 4.0](LICENSE.md); third-party
materials retain their respective licenses.

## Read without the training repository

Use Python 3.10 or later. Run the following commands from the **downloaded
dataset root**, which contains `dataset.json` and `readers/`. The copy of this
guide under the code repository's `docs/dataset/` documents the downloaded package.

```bash
cd /path/to/EgoEMG_dataset
python -m pip install -r readers/requirements.txt
python readers/inspect_dataset.py --root .
python readers/inspect_dataset.py --root . --all-sessions --media
```

The first command-line inspection opens a representative session from each
source. `--all-sessions` visits every session; `--media` additionally decodes
RGB, an available crop, depth and a mask where present. It samples media from each session. The EMG and angle reader requires
only NumPy. Other dependencies are imported when their modalities are read.

```python
import sys
from pathlib import Path
root = Path('/path/to/EgoEMG_dataset')
sys.path.insert(0, str(root / 'readers'))
from egoemg_reader import Dataset

data = Dataset(root)
session = data.session(next(iter(data.entries)))
hand = next(k for k, v in session.metadata['hands'].items() if v)
emg = session.array(hand, 'emg_filtered')[:2000]  # [samples,8], mV
angles, valid = session.angles(hand, 0, 2000)     # [samples,22], radians
roles = session.evaluation_roles(hand, 0, 2000)   # 0 train, 1 val, 2 test, -1 excluded
beta = session.mano_beta(hand, 0)                # [10], correct recording layout
```

See [SCHEMA.md](SCHEMA.md) for units, angle order, time axes, masks, channel
order, camera-frame indexing and MANO conventions. No MANO model is needed
to read the stored parameters.

## Dataset layout

- `dataset.json`: session list, counts, vocabularies, units and policy paths.
- `sessions/<session>/session.json`: active arrays and media for one session.
- `assets/egoemg_emg_normalization_filtered.json` and
  `egoemg_emg_normalization_raw.json`: training-set EMG statistics in mV.
- `calibrations/`: complete camera intrinsics, extrinsics and sensor metadata.
- `objects/`: object registry, available meshes, textures and reference photographs.
- `supplements/`: manifest for the available SAM3 mask annotations.
- `readers/`: standalone readers; the depth plugin is included from V2 onward.
- `RELEASE_MANIFEST.json`: version, file sizes, and available SHA-256 hashes.
- `UPLOAD_FILES.txt`: the complete file list for the current version.
- `UPLOAD_CHANGES.txt`: publisher upload list relative to the preceding snapshot.
  Training caches are excluded.

Storage format/schema numbers describe file encoding. Existing payload filenames are stable storage identifiers.

## Training, validation and test

`split_id` identifies generalization categories. Resolve training, validation
and test roles using the bundled evaluation protocol or `session.evaluation_roles(...)`:

- `split_id == 0`: train.
- `split_id` 1/2/3: user/gesture/both holdouts, divided by session into val/test.
- `split_id == 4`: object holdout; cleanser bottle (object 8) belongs to
  validation and purple paper box (object 0) to test.
- `split_id == -1`: excluded.

Fit on train, select models on validation, and evaluate test after selection.
Use the bundled training normalization statistics for all roles.

## Version compatibility

The session archive differs from earlier flat episode memmaps and preview shards.
Use the dataset layout and normalization statistics expected by the checkpoint.

## Downloading and relocating

Download the complete public directory, preserving relative paths. When updating
an existing local copy, merge newly added files and replace updated metadata.
Avoid reading a copy while an upgrade is in progress. The publisher updates
`dataset.json` and `RELEASE_MANIFEST.json` after payload uploads finish, then
announces the version on GitHub.

For a reading check, use `readers/inspect_dataset.py` from the downloaded
dataset root as shown above. `RELEASE_MANIFEST.json.verification`
describes the available package checks. Payload SHA-256 checks require
publisher-provided hashes; matching file sizes alone does not verify content.

V2 depth uses lossless RVL + Zstd, on a timeline independent of RGB. V2 includes
`readers/depth_rvl/README.md` with the native reader build instructions. With
the declared calibration and coordinate conventions, depth can be used for
3D backprojection and cross-camera registration where those calibrations apply.

Dataset terms are in [LICENSE.md](LICENSE.md); software has separate license notices.
