# Visualization Guide — EgoEMG

This document covers all visualization capabilities in the EgoEMG codebase, organized by **output format** and **use case**.

## Quick Start

| Goal | Script / Module | Output |
|------|----------------|--------|
| **Unified dataset viz** (4 modes) | `scripts/viz/visualize_dataset.py` | see below |
| MANO inference results viz | `scripts/viz/viz_mano_results.py` | `.glb` |
| Batch IK result comparison | `scripts/viz/visualize_ik_results.py` | `.glb` / `.png` |
| Ninapro angle animation | `scripts/viz/visualize_ninapro_angles.py` | `.html` |
| PiMforce per-joint plots | `scripts/viz/visualize_pimforce_joints.py` | `.png` |
| EMG / Optuna diagnostics | `scripts/viz/plot_*.py` | `.png` |

---

## Three Visualization Pipelines

There are three distinct pipelines that convert model predictions or ground truth into visual output:

### Pipeline A: Joint Angles → UmeTrack FK → Plotly 3D Mesh (HTML)

**Input:** 22-dim joint angles (radians) — 20 finger + 2 wrist
**Output:** Interactive HTML with Play/Pause/Slider controls

```
joint_angles (B,C,T) → skin_vertices() (UmeTrack LBS) → plotly.Mesh3d → animate_frames()
```

**Key module:** `egoemg/visualization.py`

**Core functions:**
- `get_plotly_animation_for_joint_angles(joint_angles, ...)` — end-to-end: angles → animated Plotly figure
- `skin_mesh_from_angles(joint_angles, user_profile, flip)` — single frame: angles → (vertices, triangles)
- `joint_angles_to_frames(joint_angles_t, ...)` — batched: angles → numpy frame arrays
- `joint_angles_to_frames_parallel(joint_angles, ...)` — parallel version with `ProgressParallel`
- `fig_to_array(fig)` — renders Plotly figure to numpy image array

**Used by:**
- `scripts/viz/visualize_ninapro_angles.py` — Ninapro glove angle animation
- `egoemg/lightning.py` — `LandmarkDistances` metric uses forward kinematics for fingertip distance

**Dependencies:** `plotly`, `egoemg.UmeTrack` (hand_skinning, HandModel)

---

### Pipeline B: MANO Parameters → manotorch → GLB

**Input:** `mano_pose` (48-dim axis-angle) + `mano_beta` (10-dim shape)
**Output:** GLB file viewable in [glTF Viewer](https://gltf-viewer.donmccurdy.com/) or Blender

```
mano_pose (48) + mano_beta (10) → manotorch.ManoLayer → verts (778,3) + faces
→ Kabsch alignment → trimesh.Trimesh + icosphere markers → scene.export(.glb)
```

**Key files:**
- `scripts/viz/visualize_dataset.py mesh --glb-only` — GT MANO GLBs from dataset (no video)
- `scripts/viz/viz_mano_results.py` — model inference results
- `scripts/mano/infer_mano_for_egoemg.py` — MANO parameter generation

**Core logic (in `egoemg/visualization/viz_utils.py`):**
- `ManoMeshDecoder.decode(pose, beta, hand)` → verts (778,3), faces, 21 surface markers
- `save_glb_with_markers(...)` → trimesh scene export

**21 surface markers** (MANO vertex indices): `[191, 88, 253, 708, 729, 144, 87, 295, 319, 220, 365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673]`

**Kabsch alignment:** Per-frame SVD-based alignment between predicted MANO surface markers and GT mocap keypoints. Pre-computed transforms are stored per-episode as `*_world_R.npy` / `*_world_t.npy` and synced into memmap as `mocap_mano_{hand}_world_transform`.

**Dependencies:** `manotorch`, `trimesh`, `torch`

---

### Pipeline C: MANO Parameters + Precomputed World Transform → 2D Video Overlay

**Input:** MANO pose/beta + precomputed hand-specific local-to-world transform + camera intrinsics + mocap camera transform
**Output:** MP4 video or PNG frames with mesh overlaid on webcam frames

```
mano_pose + beta → smplx.MANO → verts_local (778,3)
left hand: verts_local = mirror_x(verts_local)
verts_world = verts_local @ world_R^T + world_t
→ cv2.projectPoints → verts_2d (pixel coords) + verts_cam (camera frame)
→ map calib resolution (3840×3360) to raw video resolution
→ composite on RGB frame → cv2.VideoWriter / PNG
```

**Projection pipeline:**
1. Decode raw MANO vertices from `generated_mano_{hand}_pose` and `generated_mano_{hand}_beta`
2. For left hand, mirror raw MANO geometry along x to recover left-hand chirality
3. Load the matching per-frame local-to-world transform (`mocap_mano_{hand}_world_transform` or synced `*_world_R.npy` / `*_world_t.npy`)
4. Apply `verts_world = verts_local @ R^T + t`
5. Read `mocap_head_transform` as `T_W_Camera`, then invert to `T_C_W`
6. `cv2.projectPoints(points_world, rvec, tvec, K_calib, dist_coeffs)`
7. Map calib resolution (3840×3360) to raw video resolution (1280×720)

**Key files:**
- `scripts/viz/visualize_dataset.py mesh` — offline mesh overlay generation
- `scripts/mano/compute_mano_world_transforms.py` — precompute `*_world_R.npy` / `*_world_t.npy`
- `scripts/update_mano_world_transform_memmap.py` — sync those transforms into memmap

**Important:** `world_R/world_t` are computed against raw MANO vertices after the hand-specific chirality fix. Do not subtract the MANO wrist joint before applying them.

**Dependencies:** `smplx`, `torch`, `cv2`

---

## Script Details

### `scripts/viz/visualize_dataset.py` — Unified dataset visualization

Single multi-function entrypoint for visualizing the dataset itself
(ground truth only, no model inference).  Pick a mode with the first
argument; every mode accepts the shared options
(`--memmap-dir`, `--data-root`, `--allintra-root`, `--output-dir`,
`--device`, `--seed`).

```bash
# vision: video replay with MANO/FK mesh projection,
# mocap markers and per-hand bboxes overlaid -> MP4
python scripts/viz/visualize_dataset.py vision --episode-id episode_000000 \
    --stride 10 --max-frames 300

# timeline: EMG / joint angles / MANO multi-panel time series -> PNG
python scripts/viz/visualize_dataset.py timeline --episode 3 --hand right --window 2000

# mesh: MANO/FK mesh overlay on head-view frames -> PNG + GLB + occlusion metrics
python scripts/viz/visualize_dataset.py mesh --n-samples 10 --render-mode mesh

# mesh --glb-only: GT MANO + FK world-space GLBs with mocap/MANO-surface
# markers, no videos needed (supersedes the former `mano` mode)
python scripts/viz/visualize_dataset.py mesh --glb-only --n-samples 10

# fk_vs_mano: UmeTrack FK vs MANO mesh comparison -> GLB
python scripts/viz/visualize_dataset.py fk_vs_mano --num-samples 10

```

`vision` reads the unified memmap, so it supports both EgoEMG and ShowEE
episodes as long as the selected episode has a corresponding all-intra video.

The world-space MANO mesh path (used by the `mesh` mode) is:

1. Decode raw MANO vertices from `generated_mano_{hand}_pose` + beta
2. For left hand, mirror raw MANO geometry along x
3. Apply per-frame `mocap_mano_{hand}_world_transform`
4. Project with `mocap_head_transform` and calibration intrinsics

Do not use wrist pose/orientation to place the mesh in world coordinates.
**GPU rendering (pyrender EGL):** `--render-mode mesh` uses pyrender and
prefers the EGL (GPU) backend. EGL needs access to `/dev/dri` nodes —
grant it once (needs sudo), then set `PYOPENGL_PLATFORM=egl`:

```bash
# The egoemg_env activation hook selects the host GLVND dispatcher,
# which is required on this machine for NVIDIA EGL rendering.
conda activate egoemg_env

# permanent: add your user to the video/render groups (re-login to apply)
sudo usermod -aG video,render $USER
# immediate effect in the current session (no re-login needed):
sudo setfacl -m u:$USER:rw /dev/dri/card* /dev/dri/renderD*

python scripts/viz/visualize_dataset.py vision \
    --episode-id episode_000000 --render-mode mesh ...
```

Verify it is really on the GPU (RTX 4090, not osmesa software fallback):

```bash
conda activate egoemg_env
python -c "
from OpenGL import GL
import numpy as np, pyrender as pr
r = pr.OffscreenRenderer(128, 128)
print(GL.glGetString(GL.GL_RENDERER))   # -> b'NVIDIA GeForce RTX 4090/PCIe/SSE2'
r.delete()"
```

The script patches pyrender's EGL device-enumeration bindings (a PyOpenGL
3.1.10 omission) and falls back to osmesa (software, identical output) with
a `[pyrender]` message when EGL is unavailable.

MANO model path: explicit `--mano-model-path` > `$EGOEMG_ROOT/data/mano_data/models`.

### `scripts/viz/viz_mano_results.py` — MANO Inference Results Viz

Visualizes results from the WiLoR/Markers2MANO pipeline. Loads parquet episodes with GT keypoints, runs `EfficientGraphTransformer` model inference, Kabsch-aligns, exports GLB.

**GLB contents:**
- Green mesh: predicted MANO
- Gold spheres: predicted surface markers
- Red spheres: GT mocap keypoints

**Output:** `data/EgoEMG/mano_viz/chunk-000/viz_{ckpt}_ep{episode}/`

**Dependencies:** `manotorch`, `markers2mano`, `trimesh`, `pyarrow`

---



## Dependencies

| Script / Module | Required packages |
|----------------|-------------------|
| `visualize_dataset.py` | `cv2`, `numpy`, `trimesh`, `matplotlib`, `smplx`, `pyrender` (lazy) |
| `viz_mano_results.py` | `trimesh`, `manotorch`, `markers2mano`, `pyarrow` |
| `visualize_ik_results.py` | `trimesh`, `manotorch`, `pyarrow` |
| `visualize_ninapro_angles.py` | `plotly`, `scipy.io` |
| `visualize_pimforce_joints.py` | `matplotlib` |
| `plot_emg_value_distribution.py` | `matplotlib`, `numpy` |
| `plot_optuna_val_mae_curve.py` | `matplotlib`, `optuna` |
| `egoemg/visualization.py` | `plotly`, UmeTrack, `joblib`, `PIL` |
| `egoemg/visualization/egoemg_vis.py` | `smplx`, `cv2`, `numpy` |
| `egoemg/visualization/mesh_renderer.py` | `cv2` only |
| `egoemg/kinematics.py` | `torch`, UmeTrack |

---

## Data Layout Reference

### Dataset fields

```python
# EgoEmgMemmapDataset (with mano modality)
sample = {
    "emg": (T, 16),              # EMG signals
    "joint_angles": (T, 22),     # 20 finger + 2 wrist
    "mano_pose": (T, 48),        # MANO axis-angle (16 joints × 3)
    "mano_beta": (10,),           # MANO shape
    "mano_trans": (3,),           # mean Kabsch translation
    "label_valid_mask": (T, 22), # validity mask
}

# EgoEMG visualizer (webcam video overlay)
# Requires: mocap_webcam_transform, generated_mano_{hand}_pose,
#           generated_mano_{hand}_beta, and *_world_R.npy / *_world_t.npy
```

### MANO → 21 surface markers

Vertex indices: `[191, 88, 253, 708, 729, 144, 87, 295, 319, 220, 365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673]`

### Joint ordering (22-dim)

- Joints 0-19: finger joint angles (emg2pose ordering, see `egoemg/constants.JOINTS`)
- Joints 20-21: wrist angles (2-DoF: flexion/extension + radial/ulnar deviation)
