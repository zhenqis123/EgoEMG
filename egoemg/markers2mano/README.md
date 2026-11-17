# Markers2MANO inference

Markers2MANO accepts 21 ordered surface markers and predicts canonical
MANO-right pose and shape for either hand. Use the pretrained model to convert marker sequences with the command line or Python API.

[Download the checkpoint bundle](https://pan.baidu.com/s/1DMgtNnJdeTZNh1V975IBBQ?pwd=eg8m) (Baidu Netdisk code: `eg8m`)
and select `markers2mano/markers2mano.ckpt`.

Follow the [repository setup](../../README.md#setup), then place the checkpoint at `$EGOEMG_ROOT/pretrained_models/markers2mano/markers2mano.ckpt`
(or under the source checkout when `EGOEMG_ROOT` is unset).
Conversion uses the project's PyTorch, NumPy, SciPy and NetworkX dependencies.
MANO model assets and `manotorch` are only needed for downstream mesh decoding.

## Input format

Use a NumPy NPZ containing `markers_m` with shape `(N,21,3)`, in metres. This is
the surface-marker layout used by this model; anatomical joint keypoints use a
different layout. The marker order is:

| Indices | Names |
|---|---|
| 0 | `HandM` |
| 1–4 | `HandThumb`, `ThumbM1`, `ThumbM2`, `ThumbM3` |
| 5–8 | `HandIndex`, `IndexM1`, `IndexM2`, `IndexM3` |
| 9–12 | `HandMiddle`, `MiddleM1`, `MiddleM2`, `MiddleM3` |
| 13–16 | `HandRing`, `RingM1`, `RingM2`, `RingM3` |
| 17–20 | `HandPinky`, `PinkyM1`, `PinkyM2`, `PinkyM3` |

Optional NPZ fields are `valid_mask` of shape `(N,21)` or `(N,)`, scalar
`handedness` (`left` or `right`), and `timestamps_s` / `source_rows` of shape
`(N,)`. The latter two arrays pass through to the output. `--hand` can supply
handedness when the file omits it. An `(N,21,3)` NPY is also accepted with
an explicit `--hand`.

## Convert

From the repository root:

```bash
python -m egoemg.markers2mano \
  --checkpoint "$EGOEMG_ROOT/pretrained_models/markers2mano/markers2mano.ckpt" \
  --input /path/to/markers.npz --hand right \
  --output logs/conversion/mano.npz --device cuda:0
```

Use `--device cpu` for CPU inference and `--batch-size` to control memory use.
The default checkpoint is under `$EGOEMG_ROOT/pretrained_models/markers2mano/`
when that environment variable is set, or under the source checkout otherwise.

For repeated Python calls, load the model once:

```python
import os
from pathlib import Path
import numpy as np
from egoemg.markers2mano import load_model, convert_markers

checkpoint = Path(os.environ.get("EGOEMG_ROOT", ".")) / "pretrained_models/markers2mano/markers2mano.ckpt"
model = load_model(checkpoint, "cuda:0")
markers = np.load("markers.npy")
result = convert_markers(model, markers, handedness="right", batch_size=32)
```

## Output and coordinates

| Field | Shape | Meaning |
|---|---|---|
| `mano_pose` | `(N,48)` | Axis-angle pose in radians; the root rotation is zero |
| `mano_beta` | `(N,10)` | Per-frame shape predictions |
| `mano_beta_mean` | `(10,)` | Mean shape over valid converted frames |
| `valid` | `(N,)` | Complete, finite, nondegenerate frames with finite predictions |
| `world_to_canonical` | `(N,4,4)` | Input world markers → normalized network input |

Both hands use MANO-right semantics with `flat_hand_mean=False`. Input
normalization uses the five-marker palm origin and marker-defined axes;
left inputs additionally reflect local Z. Incomplete, nonfinite or degenerate
frames receive `valid=False` and NaN parameters. Conversion preserves input
rows and performs no temporal interpolation.

These are network predictions before downstream MANO/UmeTrack fitting. The
`world_to_canonical` matrix describes the marker normalization frame. Global
MANO mesh placement requires a separate rigid fit; left-hand world geometry
uses the original production convention of reflecting MANO X before that fit.
The batched rigid-fit helper is available in `egoemg.markers2mano.rigid_align`.
