# Visualization Guide — EMG2Pose

This document covers all visualization capabilities in the EMG2Pose codebase, organized by **output format** and **use case**.

## Quick Start

| Goal | Script / Module | Output |
|------|----------------|--------|
| MANO mesh GLB from dataset | `scripts/viz/viz_mano_from_dataset.py` | `.glb` |
| Multi-modal time-series PNG | `scripts/viz/viz_emg_pose_timeline.py` | `.png` |
| MANO inference results viz | `scripts/viz/viz_mano_results.py` | `.glb` |
| MANO mesh overlay on webcam video | `scripts/viz/visualize_egoemg_mesh.py` | `.mp4` / `.png` |
| EgoEMG vision dataset sample debug | `scripts/viz/visualize_egoemg_vision_dataset.py` | `.png` |
| Plotly 3D hand mesh animation | `egoemg/visualization.py` | `.html` |
| EMG2Pose / PiMforce interactive viz | `scripts/visualization/visualize_emg2pose_dataset.py` | `.html` |
| Ninapro angle animation | `scripts/visualization/visualize_ninapro_angles.py` | `.html` |
| PiMforce per-joint plots | `scripts/visualization/visualize_pimforce_joints.py` | `.png` |
| Paper figures | `scripts/paper/generate_paper_figures.py` | `.pdf` / `.png` |

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
- `scripts/visualization/visualize_emg2pose_dataset.py` — interactive dataset viz with optional model inference
- `scripts/visualization/visualize_ninapro_angles.py` — Ninapro glove angle animation
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
- `scripts/viz/viz_mano_from_dataset.py` — GT MANO from dataset
- `scripts/viz/viz_mano_results.py` — model inference results
- `scripts/mano/infer_mano_for_egoemg.py` — MANO parameter generation

**Core functions (in `viz_mano_from_dataset.py`):**
- `decode_mano(pose, beta, device)` → verts (778,3), faces, 21 surface markers
- `save_glb(out_path, verts, faces, gt_markers, pred_markers)` → trimesh scene export

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
5. Read `mocap_webcam_transform` as `T_W_Camera`, then invert to `T_C_W`
6. `cv2.projectPoints(points_world, rvec, tvec, K_calib, dist_coeffs)`
7. Map calib resolution (3840×3360) to raw video resolution (1280×720)

**Key files:**
- `scripts/viz/visualize_egoemg_mesh.py` — offline video generation
- `scripts/mano/compute_mano_world_transforms.py` — precompute `*_world_R.npy` / `*_world_t.npy`
- `scripts/update_mano_world_transform_memmap.py` — sync those transforms into memmap

**Important:** `world_R/world_t` are computed against raw MANO vertices after the hand-specific chirality fix. Do not subtract the MANO wrist joint before applying them.

**Dependencies:** `smplx`, `torch`, `cv2`

---

## Script Details

### `scripts/viz/viz_mano_from_dataset.py` — MANO 3D Mesh GLB

Loads `EgoEmgMemmapDataset` samples with MANO modality, decodes to mesh, Kabsch-aligns to world frame, exports GLB.

```bash
# Single episode, right hand, 2 frames
python scripts/viz/viz_mano_from_dataset.py --episode 3 --hand right --num-frames 2

# Multiple episodes, 3 frames each
python scripts/viz/viz_mano_from_dataset.py --episodes 0 3 10 20 --num-frames 3

# All episodes, 1 frame each
python scripts/viz/viz_mano_from_dataset.py --all --num-frames 1

# Use GPU
python scripts/viz/viz_mano_from_dataset.py --episode 3 --hand left --device 0
```

**GLB contents:**
- Green semi-transparent mesh: MANO hand model
- Gold spheres: GT mocap markers (21 points)
- Blue spheres: MANO-predicted surface markers

**Output:** `data/EgoEMG/mano_viz/dataset_samples/`

**Prerequisites:**
```bash
pip install trimesh manotorch
# MANO assets: ../HandVQVAE/assets/mano
```

---

### `scripts/viz/viz_emg_pose_timeline.py` — Multi-Modal Timeline PNG

Generates multi-panel matplotlib figure with vertically aligned time-series:
1. EMG signals (16 channels, filtered)
2. Joint angles (22-dim: 20 finger + 2 wrist)
3. MANO pose (16 joints, rotation magnitude)

```bash
# Default: episode 3, right hand, offset=100000, window=2000 frames
python scripts/viz/viz_emg_pose_timeline.py

# Custom parameters
python scripts/viz/viz_emg_pose_timeline.py --episode 10 --hand left --offset 50000 --window 5000

# Custom output path
python scripts/viz/viz_emg_pose_timeline.py --episode 3 --hand right --out-path /tmp/timeline.png
```

**Output:** `data/EgoEMG/mano_viz/timelines/`

---

### `scripts/viz/viz_mano_results.py` — MANO Inference Results Viz

Visualizes results from the WiLoR/Markers2MANO pipeline. Loads parquet episodes with GT keypoints, runs `EfficientGraphTransformer` model inference, Kabsch-aligns, exports GLB.

**GLB contents:**
- Green mesh: predicted MANO
- Gold spheres: predicted surface markers
- Red spheres: GT mocap keypoints

**Output:** `data/EgoEMG/mano_viz/chunk-000/viz_{ckpt}_ep{episode}/`

**Dependencies:** `manotorch`, `markers2mano`, `trimesh`, `pyarrow`

---

### `scripts/viz/visualize_egoemg_mesh.py` — MANO Mesh on Webcam Video

Overlays GT MANO mesh onto EgoEMG webcam frames. The only correct world-coordinate path is:

1. Decode raw MANO vertices from `generated_mano_{hand}_pose` and `generated_mano_{hand}_beta`
2. For left hand, mirror raw MANO geometry along x
3. Apply precomputed per-frame `mocap_mano_{hand}_world_transform` (or synced `*_world_R.npy` / `*_world_t.npy`)
4. Project with `mocap_webcam_transform` and calibration intrinsics

Do not use wrist pose/orientation to place the mesh in world coordinates.

```bash
python scripts/viz/visualize_egoemg_mesh.py \
    --memmap_dir data/EgoEMG_memmap \
    --data_root data/EgoEMG \
    --output ./egoemg_mesh_debug_world \
    --start_frame 100000 --n_frames 1 \
    --hand right --device cpu --save_images \
    --draw_points --draw_wireframe --draw_keypoints
```

**Output:** MP4 video or PNG frames

**Correct transform usage:**

```python
verts_local = smplx_mano(...).vertices[0].cpu().numpy()  # raw MANO verts
if hand == "left":
    verts_local[:, 0] *= -1.0
verts_world = verts_local @ world_R[frame_idx].T + world_t[frame_idx]
```

Do not wrist-center `verts_local` before applying `world_R/world_t`.

---

### `scripts/viz/visualize_egoemg_vision_dataset.py` — EgoEMG Vision Dataset Debug

Visualizes the actual samples produced by
`egoemg.datasets.egoemg_vision_dataset.EgoEmgVisionDataset`.

This is different from `visualize_egoemg_mesh.py`:

- `visualize_egoemg_mesh.py` validates the world-space MANO mesh projection path.
- `visualize_egoemg_vision_dataset.py` validates the training sample emitted to WiLoR.

For each selected dataset index it writes a side-by-side PNG:

1. Dataset-aligned raw frame
2. Denormalized training patch

The raw-frame panel shows:
- left-hand mirroring exactly as used by the dataset
- `orig_keypoints_2d`
- `orig_markers_2d`
- training bbox

The patch panel shows:
- denormalized `img`
- `keypoints_2d` mapped back from normalized patch coordinates to patch pixels

```bash
python scripts/data/build_egoemg_vision_index.py \
    --memmap-dir data/EgoEMG_memmap \
    --output-dir data/EgoEMG_memmap/vision_index

python scripts/viz/visualize_egoemg_vision_dataset.py \
    --memmap-dir data/EgoEMG_memmap \
    --video-root data/EgoEMG \
    --allintra-root data/EgoEMG_allintra \
    --vision-index-dir data/EgoEMG_memmap/vision_index \
    --output-dir /tmp/egoemg_vision_dataset_viz \
    --num-samples 16 \
    --target-hand both
```

Useful when checking:
- left/right canonicalization
- raw-frame 2D supervision
- bbox selection
- patch extraction and normalization

Important:
- this script reads webcam frames from all-intra re-encoded videos only
- decoding uses `decord` only
- missing all-intra files are treated as errors; there is no OpenCV fallback
- dataset startup requires the sidecar vision index; build it once with
  `scripts/data/build_egoemg_vision_index.py`

For the full EgoEMG vision dataset and WiLoR fine-tuning workflow, including
training commands and config overrides, see `docs/egoemg_wilor_training.md`.

---

### `scripts/visualization/visualize_emg2pose_dataset.py` — Interactive EMG2Pose Viz

Loads EMG2Pose or PiMforce session data, optionally runs model inference, generates Plotly 3D mesh animations for GT and predictions side by side.

```bash
python scripts/visualization/visualize_emg2pose_dataset.py \
    --experiment tracking_vemg2pose \
    --checkpoint /path/to/checkpoint.ckpt \
    --session-dir /path/to/session
```

**Output:** HTML file with interactive 3D hand mesh animation

---

### `scripts/visualization/visualize_ninapro_angles.py` — Ninapro Angle Viz

Maps Ninapro glove joint angles to emg2pose's 20-dim joint ordering (with sign/offset corrections), converts degrees to radians, generates Plotly 3D mesh animation.

**Key mappings:** `NINAPRO_TO_EMG2POSE` (20 entries), `EMG2POSE_SIGN`, `EMG2POSE_OFFSET_DEG`

**Output:** HTML file with 3D hand mesh animation

---

### `scripts/visualization/visualize_pimforce_joints.py` — PiMforce Per-Joint Plots

Generates per-joint matplotlib time-series plots. Each of the 20 joints gets a separate PNG showing angle over time.

**Output:** `./pimforce_visualizations/` — one PNG per joint

---

## Module API Reference

### `egoemg/visualization.py`

Meta-origin module providing UmeTrack-based hand mesh visualization via Plotly.

```python
# End-to-end animation
from egoemg.visualization import get_plotly_animation_for_joint_angles

fig = get_plotly_animation_for_joint_angles(
    joint_angles,       # (T, 22) or (B, C, T) tensor/array
    flip=False,         # True for left hand
    opacity=0.8,
    hand_model=None,    # None → loads default
    title="Hand Animation"
)
fig.write_html("output.html")
```

```python
# Batched conversion to numpy frames (for video)
from egoemg.visualization import joint_angles_to_frames

frames = joint_angles_to_frames(joint_angles, color="blue", flip=False)
# frames: (T, H, W, 4) RGBA
```

```python
# Forward kinematics (joint angles → 3D landmarks)
from egoemg.kinematics import forward_kinematics

landmarks = forward_kinematics(joint_angles)  # (..., 21, 3)
```

### Correct EgoEMG Mesh Projection

For EgoEMG MANO overlay, the repository-standard path is:

```python
from egoemg.visualization import EgoEmgVisualizer

vis = EgoEmgVisualizer(data_root, mano_model_path)
overlay = vis.render_frame(
    image,          # (H, W, 3) uint8 RGB
    pose_aa,        # (48,) MANO axis-angle
    beta,           # (10,) shape
    world_R,        # (3, 3) precomputed per-frame world rotation
    world_t,        # (3,) precomputed per-frame world translation
    T_W_Camera,     # (4, 4) from mocap_webcam_transform
    hand="right",
    color=(0.4, 0.7, 1.0),
    alpha=0.6,
)
```

`world_R/world_t` come from `scripts/mano/compute_mano_world_transforms.py`.
For left hand, they are fit on x-mirrored raw MANO marker vertices, so any extra recentering or a second mirror will create a visible projection offset.

---

## Vision-to-Pose Visualization

The vision-to-pose baseline (`Vision2PoseModule` in `egoemg/models/vision2pose.py`) outputs **joint angles** (B, T, num_joints), the same format as EMG baselines. This means it can be visualized through **Pipeline A** (Plotly animation) directly.

### How to visualize vision-to-pose predictions

1. **Run inference and save predictions:**
   ```python
   # After training, load checkpoint and run test
   from egoemg.models.vision2pose import Vision2PoseModule
   model = Vision2PoseModule.load_from_checkpoint(ckpt_path)
   pred_angles = model(images)  # (B, T, num_joints)
   ```

2. **Generate Plotly animation** (same as EMG baseline):
   ```python
   from egoemg.visualization import get_plotly_animation_for_joint_angles

   fig = get_plotly_animation_for_joint_angles(pred_angles[0], flip=False)
   fig.write_html("vision2pose_pred.html")
   ```

3. **Compare with GT MANO** (convert joint angles back to MANO mesh):

   The `Vision2PoseModule` only predicts joint angles. To get a MANO mesh from predictions, you need to convert joint angles → MANO pose. This requires an IK-style solve (see `batch_ik_mesh.py`). Alternatively, you can:
   - Use `forward_kinematics()` to get 3D landmarks (21 keypoints) for qualitative comparison
   - Use `LandmarkDistances` metric to compare predicted vs GT landmark positions
   - For full MANO mesh visualization, run the predicted angles through the same FK pipeline

### MANO projection from vision-to-pose

If you want to project MANO mesh from the WiLoR backbone's original output (which includes MANO pose/shape parameters) alongside the regression head's joint angle predictions:

1. WiLoR's backbone produces features used by both the MANO head (original) and the regression head (ours)
2. To visualize the original WiLoR MANO prediction, you would need to attach WiLoR's MANO head to the backbone features
3. For the joint angle head, use Pipeline A (UmeTrack FK → Plotly) — no MANO needed

---

## MANO Inference Pipeline (Markers2MANO)

The `EfficientGraphTransformer` model predicts MANO parameters from 21 mocap markers.

### Full pipeline (`scripts/mano/infer_mano_for_egoemg.py`)

```
Parquet episode → mocap keypoints (21,3) → root at wrist → local coords
→ EfficientGraphTransformer → pose_6d (16,6) + beta (10)
→ 6D → axis-angle → ManoLayer → verts (778,3)
→ Extract 21 surface markers → Kabsch alignment → save .npy
```

### Left-hand strategy

Both hands use MANO-right canonical parameterization. For world-space mesh visualization, left hand is recovered by mirroring raw MANO geometry along x before world alignment; face winding is then reversed for rendering. This is separate from the model-side `flip_local_z` convention used when canonicalizing mocap keypoints for MANO inference.

### Output files

```
data/EgoEMG/mano/chunk-000/
├── episode_XXXXXX_left_pose.npy    — (T, 48) axis-angle
├── episode_XXXXXX_left_beta.npy    — (10,) shape
├── episode_XXXXXX_left_trans.npy   — (3,) mean Kabsch translation
├── episode_XXXXXX_left_world_R.npy — (T, 3, 3) per-frame Kabsch rotation
├── episode_XXXXXX_left_world_t.npy — (T, 3) per-frame Kabsch translation
├── episode_XXXXXX_right_pose.npy   — same for right hand
├── ...
└── generation_report.txt           — per-episode Kabsch errors (mm)
```

### MANO angle conversion

MANO-to-UmeTrack angle conversion is done via optimization-based IK. See:

- `scripts/ik/test_ik_mesh_single.py` — single-frame IK test
- `scripts/ik/batch_ik_mesh.py` — batch IK across all frames (multi-GPU)

---

## Dependencies

| Script / Module | Required packages |
|----------------|-------------------|
| `viz_mano_from_dataset.py` | `trimesh`, `manotorch`, `torch` |
| `viz_mano_results.py` | `trimesh`, `manotorch`, `markers2mano`, `pyarrow` |
| `viz_emg_pose_timeline.py` | `matplotlib`, `numpy` |
| `visualize_egoemg_mesh.py` | `cv2`, `smplx`, `numpy` |
| `visualization/visualize_emg2pose_dataset.py` | `plotly`, Hydra, Lightning |
| `visualization/visualize_ninapro_angles.py` | `plotly`, `scipy.io` |
| `visualization/visualize_pimforce_joints.py` | `matplotlib` |
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
