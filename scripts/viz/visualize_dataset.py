#!/usr/bin/env python3
"""Unified dataset-visualization entrypoint.

Replaces the per-mode dataset viz scripts with one multi-function command::

    python scripts/viz/visualize_dataset.py <mode> [mode options]

Modes (dataset itself, ground truth only):

  vision      EgoEmgVisionDataset samples (raw frame + patch panel) -> PNG
  timeline    EMG / joint angles / MANO multi-panel time series -> PNG
  mano        GT MANO mesh + mocap markers -> GLB (Kabsch-aligned)
  mesh        MANO/FK mesh overlay on head-view frames -> PNG + GLB +
              occlusion metrics (pyrender)
  markers     mocap marker reprojection over a full episode video -> MP4
  crops       pre-cropped hand patches grid from per-episode LMDB -> JPG
  fk_vs_mano  UmeTrack FK vs MANO mesh comparison -> GLB
  align       ShowEE session frame-alignment check (markers overlay) -> PNG

Shared options (available in every mode):

    --memmap-dir      data/EgoEMG_unified_memmap
    --data-root       data
    --allintra-root   data/EgoEMG_allintra
    --output-dir      <mode-specific default under /tmp/egoemg_viz/>
    --device          cuda (cpu/cuda)
    --seed            42

Heavy dependencies (torch/smplx/pyrender/plotly/decord/lmdb) are imported
lazily inside each mode, so light modes start fast.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Importing egoemg.visualization executes its __init__ which eagerly
# imports matplotlib/plotly/torch; set the headless backend BEFORE that
# so backend probing never hangs on a headless box.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emg2pose_viz_runtime" / "mpl"))

from egoemg.visualization import viz_utils as vu  # noqa: E402

# ── vision mode ─────────────────────────────────────────────────────────────

JOINT_COLOR_BGR = (0, 220, 0)
MARKER_COLOR_BGR = (0, 255, 255)
BBOX_COLOR_BGR = (255, 180, 0)


def _denormalize_patch_rgb(img_chw: np.ndarray, mean: np.ndarray,
                           std: np.ndarray) -> np.ndarray:
    img = img_chw.astype(np.float32).copy()
    for channel_idx in range(3):
        img[channel_idx] = img[channel_idx] * float(std[channel_idx]) \
            + float(mean[channel_idx])
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def _patch_keypoints_to_pixels(keypoints_2d: np.ndarray,
                               patch_size: int) -> np.ndarray:
    out = keypoints_2d.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] + 0.5) * float(patch_size)
    out[:, 1] = (out[:, 1] + 0.5) * float(patch_size)
    return out


def _build_raw_frame_panel(dataset: Any, sample: dict, joint_radius: int,
                           marker_radius: int, bbox_line_width: int,
                           ) -> np.ndarray:
    frame_bgr = np.asarray(sample["frame_bgr"], dtype=np.uint8)
    is_mirrored = float(sample["raw_right"]) == 0.0
    video_w = frame_bgr.shape[1]

    if is_mirrored:
        frame_bgr = np.ascontiguousarray(frame_bgr[:, ::-1])

    def _unmirror_xy(pts: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        if is_mirrored:
            pts[:, 0] = (video_w - 1) - pts[:, 0]
        return pts

    panel = frame_bgr
    bbox = np.asarray(sample["bbox"], dtype=np.float32)
    if is_mirrored:
        x0, y0, x1, y1 = bbox
        bbox = np.array([(video_w - 1) - x1, y0, (video_w - 1) - x0, y1],
                        dtype=np.float32)
    panel = vu.draw_bbox(panel, bbox, BBOX_COLOR_BGR, bbox_line_width)
    panel = vu.draw_points(
        panel,
        _unmirror_xy(np.asarray(sample["orig_markers_2d"], dtype=np.float32)),
        MARKER_COLOR_BGR, marker_radius)
    panel = vu.draw_points(
        panel,
        _unmirror_xy(np.asarray(sample["orig_keypoints_2d"], dtype=np.float32)),
        JOINT_COLOR_BGR, joint_radius)

    joints_valid = int((np.asarray(sample["orig_keypoints_2d"])[:, 2] > 0).sum())
    markers_valid = int((np.asarray(sample["orig_markers_2d"])[:, 2] > 0).sum())
    lines = [
        f"dataset_idx={int(sample['_dataset_index'])} hand={sample['target_hand']}",
        f"episode={sample['episode_id']} subject={sample['episode_subject']}",
        f"frame_idx={int(sample['frame_index'])} "
        f"video_frame={int(sample['video_frame_index'])}",
        f"bbox_source={sample['bbox_source_name']} "
        f"joints={joints_valid}/21 markers={markers_valid}/21",
        f"raw_right={float(sample['raw_right']):.0f} "
        f"canonical_right={float(sample['raw_right']):.0f}",
    ]
    return vu.draw_text_block(panel, lines)


def _build_patch_panel(dataset: Any, sample: dict, joint_radius: int,
                       ) -> np.ndarray:
    import cv2
    patch_rgb = _denormalize_patch_rgb(
        np.asarray(sample["img"], dtype=np.float32), dataset.mean, dataset.std)
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    patch_points = _patch_keypoints_to_pixels(
        np.asarray(sample["keypoints_2d"], dtype=np.float32),
        int(dataset.patch_size))
    panel = vu.draw_points(patch_bgr, patch_points, JOINT_COLOR_BGR, joint_radius)
    patch_valid = int((patch_points[:, 2] > 0).sum())
    lines = [
        f"patch_size={int(dataset.patch_size)}",
        f"keypoints_2d in patch={patch_valid}/21",
        "coords reconstructed from normalized dataset output",
    ]
    return vu.draw_text_block(panel, lines)


def _make_canvas(raw_panel: np.ndarray, patch_panel: np.ndarray,
                 max_panel_width: int) -> np.ndarray:
    import cv2
    raw_h = raw_panel.shape[0]
    scale = raw_h / float(patch_panel.shape[0])
    patch_w = max(1, int(round(patch_panel.shape[1] * scale)))
    patch_resized = cv2.resize(
        patch_panel, (patch_w, raw_h), interpolation=cv2.INTER_NEAREST)
    spacer = np.full((raw_h, 24, 3), 24, dtype=np.uint8)
    canvas = np.concatenate([raw_panel, spacer, patch_resized], axis=1)
    if canvas.shape[1] <= max_panel_width:
        return canvas
    scale_out = max_panel_width / float(canvas.shape[1])
    out_w = max(1, int(round(canvas.shape[1] * scale_out)))
    out_h = max(1, int(round(canvas.shape[0] * scale_out)))
    return cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _resolve_indices(dataset_len: int, start_index: int, num_samples: int,
                     sample_indices: list[int] | None) -> list[int]:
    if sample_indices:
        indices = sample_indices
    else:
        indices = list(range(start_index, min(start_index + num_samples,
                                              dataset_len)))
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(
                f"Dataset index out of range: {idx} not in [0, {dataset_len})")
    return indices


def _requested_index_limit(start_index: int, num_samples: int,
                           sample_indices: list[int] | None) -> int:
    if sample_indices:
        return max(sample_indices) + 1
    return start_index + num_samples


def run_vision(args: argparse.Namespace) -> int:
    import cv2
    from egoemg.datasets.egoemg_vision_dataset import EgoEmgVisionDataset

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vision] Output dir: {output_dir}")
    requested_limit = _requested_index_limit(
        args.start_index, args.num_samples, args.sample_indices)
    dataset = EgoEmgVisionDataset(
        memmap_dir=Path(args.memmap_dir),
        video_root=Path(args.video_root),
        allintra_root=Path(args.allintra_root) if args.allintra_root else None,
        allintra_suffix=args.allintra_suffix,
        vision_index_dir=Path(args.vision_index_dir) if args.vision_index_dir else None,
        auto_build_index=args.auto_build_index,
        calibration_path=Path(args.calibration_path) if args.calibration_path else None,
        allowed_episode_ids=args.allowed_episode_ids,
        allowed_subjects=args.allowed_subjects,
        allowed_splits=args.allowed_splits,
        target_hand=args.target_hand,
        stride=args.stride,
        index_limit=requested_limit,
        patch_size=args.patch_size,
        do_augment=False,
        return_frame_bgr=True,
        log_init_timing=True,
    )
    indices = _resolve_indices(len(dataset), args.start_index,
                               args.num_samples, args.sample_indices)
    for render_i, dataset_idx in enumerate(indices, start=1):
        sample = dataset[dataset_idx]
        sample["_dataset_index"] = np.int64(dataset_idx)
        raw_panel = _build_raw_frame_panel(
            dataset, sample, joint_radius=args.joint_radius,
            marker_radius=args.marker_radius,
            bbox_line_width=args.bbox_line_width)
        if args.raw_only:
            canvas = raw_panel
        else:
            patch_panel = _build_patch_panel(
                dataset, sample, joint_radius=args.joint_radius)
            canvas = _make_canvas(raw_panel, patch_panel,
                                  max_panel_width=args.max_panel_width)
        out_path = output_dir / (
            f"sample_{dataset_idx:06d}_ep_{sample['episode_id']}"
            f"_frame_{int(sample['frame_index']):08d}_{sample['target_hand']}.png")
        cv2.imwrite(str(out_path), canvas)
        print(f"[{render_i}/{len(indices)}] Wrote {out_path}")
    print("[vision] Done")
    return 0


# ── timeline mode ───────────────────────────────────────────────────────────

MANO_JOINT_NAMES = [
    "wrist", "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
]


def _plot_timeline(sample: dict, ep_id: str, hand: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    has_emg = "emg" in sample
    has_ja = "joint_angles" in sample
    has_mano = "mano_pose" in sample
    n_panels = sum([has_emg, has_ja, has_mano])
    if n_panels == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels),
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    panel = 0

    if has_emg:
        ax = axes[panel]
        emg = sample["emg"]  # (C, T)
        T = emg.shape[1]
        t = np.arange(T)
        for ch in range(emg.shape[0]):
            ax.plot(t, emg[ch] + ch * 0.5, linewidth=0.3, alpha=0.8)
        ax.set_ylabel("EMG channels")
        ax.set_title(f"{ep_id} / {hand} hand — Filtered EMG ({emg.shape[0]} ch)")
        ax.set_xlim(0, T)
        panel += 1

    if has_ja:
        ax = axes[panel]
        ja = sample["joint_angles"]  # (C, T)
        T = ja.shape[1]
        t = np.arange(T)
        n_angles = ja.shape[0]
        cmap = plt.cm.tab20(np.linspace(0, 1, n_angles))
        for ch in range(n_angles):
            label = f"j{ch}" if ch < 20 else (
                "wrist_pitch" if ch == 20 else "wrist_yaw")
            ax.plot(t, ja[ch], linewidth=0.6, alpha=0.8, color=cmap[ch],
                    label=label)
        ax.set_ylabel("Joint angles (rad)")
        ax.set_title(f"Joint angles ({n_angles}-dim)")
        ax.legend(fontsize=5, ncol=6, loc="upper right")
        ax.set_xlim(0, T)
        panel += 1

    if has_mano:
        ax = axes[panel]
        pose = sample["mano_pose"]  # (T, 48)
        T = pose.shape[0]
        t = np.arange(T)
        for j in range(16):
            mag = np.linalg.norm(pose[:, j * 3:(j + 1) * 3], axis=1)
            label = MANO_JOINT_NAMES[j] if j < len(MANO_JOINT_NAMES) else f"j{j}"
            ax.plot(t, mag, linewidth=0.5, alpha=0.7, label=label)
        ax.set_ylabel("MANO joint rotation magnitude (rad)")
        ax.set_title(
            f"MANO pose (48-dim axis-angle, beta shape: "
            f"{sample.get('mano_beta', np.zeros(0)).shape})")
        ax.legend(fontsize=5, ncol=4, loc="upper right")
        ax.set_xlim(0, T)
        panel += 1

    axes[-1].set_xlabel("Frame index (within window)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_timeline(args: argparse.Namespace) -> int:
    ds = vu.make_memmap_dataset(
        memmap_dir=args.memmap_dir, hand=args.hand,
        window_length=args.window,
        modalities=["emg", "joint_angles", "mocap_hands", "mano"])
    print(f"Dataset: {len(ds)} windows, hand={args.hand}")

    win_idx = vu.find_window_at_offset(ds, args.episode, args.offset)
    if win_idx is None:
        block_mask = np.asarray(ds._block_episode_idx) == args.episode
        block_ids = np.where(block_mask)[0]
        if len(block_ids) > 0:
            win_idx = int(ds._block_cumsum[block_ids[0]])
    if win_idx is None:
        print(f"Episode {args.episode} not found in dataset.")
        return 0

    sample = ds[win_idx]
    ep_id = sample.get("episode_id", f"episode_{args.episode:06d}")
    start = sample.get("window_start_idx", 0)
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = Path(args.output_dir) / f"{ep_id}_{args.hand}_f{start}.png"
    _plot_timeline(sample, ep_id, args.hand, out_path)
    return 0


# ── mano mode ───────────────────────────────────────────────────────────────

def _process_mano_sample(ds: Any, sample_idx: int, decoder: vu.ManoMeshDecoder,
                         out_dir: Path, hand: str) -> str | None:
    sample = ds[sample_idx]
    if "mano_pose" not in sample or "mano_beta" not in sample:
        return None
    pose = sample["mano_pose"]
    beta = sample["mano_beta"]
    mid = pose.shape[0] // 2
    verts, faces = decoder.decode(pose[mid], beta, hand)
    pred_markers = decoder.marker_vertices(verts)

    gt_markers = None
    if "mocap_keypoints" in sample:
        kp = sample["mocap_keypoints"]
        if kp.ndim == 3 and kp.shape[0] > mid:
            gt_markers = kp[mid].astype(np.float32)

    if gt_markers is not None and "mano_world_R" in sample \
            and "mano_world_t" in sample:
        R = np.asarray(sample["mano_world_R"][mid], dtype=np.float64)
        t = np.asarray(sample["mano_world_t"][mid], dtype=np.float64)
        verts = verts @ R.T + t
        pred_markers = pred_markers @ R.T + t
    elif gt_markers is not None:
        R, t = vu.umeyama_alignment(pred_markers, gt_markers)
        verts = verts @ R.T + t
        pred_markers = pred_markers @ R.T + t

    ep_idx, ep_id, center = vu.window_location(ds, sample_idx)
    fname = f"{ep_id}_{hand}_frame{center:07d}.glb"
    out_path = out_dir / fname
    vu.save_glb_with_markers(out_path, verts, faces,
                             gt_markers=gt_markers,
                             pred_markers=pred_markers)
    return str(out_path)


def run_mano(args: argparse.Namespace) -> int:
    ds = vu.make_memmap_dataset(
        memmap_dir=args.memmap_dir, hand=args.hand,
        window_length=args.window,
        mano_npy_dir=args.mano_npy_dir)
    num_episodes = len(ds._episode_id)
    print(f"Dataset: {len(ds)} windows, {num_episodes} episodes, hand={args.hand}")

    if args.episode is not None:
        ep_list = [args.episode]
    elif args.episodes is not None:
        ep_list = args.episodes
    elif args.all:
        ep_list = list(range(num_episodes))
    else:
        ep_list = [0]

    decoder = vu.ManoMeshDecoder(args.mano_model_path, args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    for ep_idx in ep_list:
        win_indices = vu.find_window_indices(ds, ep_idx)
        if not win_indices:
            print(f"  episode {ep_idx}: no windows, skipping")
            continue
        if args.offset is not None and len(ep_list) == 1:
            target_win = args.offset // args.window
            idx = win_indices[min(target_win, len(win_indices) - 1)]
            path = _process_mano_sample(ds, idx, decoder, out_dir, args.hand)
            if path:
                print(f"  Saved: {path}")
                total_saved += 1
        else:
            n = min(args.num_frames, len(win_indices))
            chosen = np.linspace(0, len(win_indices) - 1, n, dtype=int)
            for c in chosen:
                path = _process_mano_sample(ds, win_indices[c], decoder,
                                            out_dir, args.hand)
                if path:
                    print(f"  Saved: {path}")
                    total_saved += 1
    print(f"\nDone. {total_saved} GLB files saved to {out_dir}")
    return 0


# ── mesh mode ───────────────────────────────────────────────────────────────

FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def _render_mesh_overlay(frame_bgr: np.ndarray,
                         hand_meshes: list[tuple[np.ndarray, np.ndarray,
                                                 tuple[int, int, int]]],
                         T_W_C: np.ndarray, K_vid: np.ndarray,
                         renderer: Any, alpha: float) -> np.ndarray:
    import cv2
    import pyrender
    import trimesh

    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    for verts, faces, color_rgb in hand_meshes:
        tm = trimesh.Trimesh(
            vertices=verts.astype(np.float32), faces=faces)
        tm.visual.vertex_colors = list(color_rgb) + [255]
        scene.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
    T_W_C_gl = T_W_C @ FLIP_YZ
    cam = pyrender.IntrinsicsCamera(
        fx=K_vid[0, 0], fy=K_vid[1, 1],
        cx=K_vid[0, 2], cy=K_vid[1, 2],
        znear=0.01, zfar=100.0)
    scene.add(cam, pose=T_W_C_gl)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=T_W_C_gl)
    color_rgb, depth = renderer.render(scene)
    mask = depth > 0
    if not mask.any():
        return frame_bgr
    overlay_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    out = frame_bgr.copy()
    mask_3 = mask[:, :, None]
    out = np.where(
        mask_3,
        (alpha * overlay_bgr + (1.0 - alpha) * frame_bgr).astype(np.uint8),
        out)
    return out


def run_mesh(args: argparse.Namespace) -> int:
    from tqdm import tqdm
    import cv2

    manifest = vu.load_manifest(args.memmap_dir)
    md = vu.load_metadata(args.memmap_dir)
    calib = vu.load_calibration(vu.resolve_calibration_path(
        args.data_root, args.calibration_path))
    K_raw, dist_raw, calib_w, calib_h = (
        calib.K, calib.dist, calib.width, calib.height)

    cam_tracked = vu.load_memmap(args.memmap_dir, manifest, "mocap_head_tracked")
    cam_transform = vu.load_memmap(args.memmap_dir, manifest, "mocap_head_transform")
    frame_idx_mm = vu.load_memmap(args.memmap_dir, manifest, "image_head_frame_index")
    ep_idx_mm = vu.load_memmap(args.memmap_dir, manifest, "episode_index")
    video_paths = vu.decode_bytes(md["episode_head_video_path"])
    beta_idx_arr = md["episode_beta_idx"]

    decoder = vu.ManoMeshDecoder(args.mano_model_path, args.device)
    faces_right = decoder._faces
    faces_left = faces_right[:, [0, 2, 1]]
    hand_faces = {"right": faces_right, "left": faces_left}

    hand_data = {}
    for hand in ("left", "right"):
        hand_data[hand] = {
            "pose": vu.load_memmap(args.memmap_dir, manifest,
                                   f"generated_mano_{hand}_pose"),
            "world": vu.load_memmap(args.memmap_dir, manifest,
                                    f"mocap_mano_{hand}_world_transform"),
            "beta": vu.load_memmap(args.memmap_dir, manifest,
                                   f"generated_mano_{hand}_beta",
                                   section="episode_fields"),
            "joint_angles": vu.load_memmap(args.memmap_dir, manifest,
                                           f"generated_joint_angles_{hand}"),
            "keypoints": vu.load_memmap(args.memmap_dir, manifest,
                                        f"mocap_{hand}_keypoints"),
            "keypoints_valid": vu.load_memmap(args.memmap_dir, manifest,
                                              f"mocap_{hand}_valid"),
        }

    rng = np.random.RandomState(args.seed)
    valid_indices = np.where(cam_tracked == 1)[0]
    n_samples = min(args.n_samples, len(valid_indices))
    sampled_indices = sorted(
        rng.choice(valid_indices, size=n_samples, replace=False))
    ep_frame_map: dict[int, list[int]] = {}
    for global_i in sampled_indices:
        ep_frame_map.setdefault(int(ep_idx_mm[global_i]), []).append(global_i)

    vrs: dict[int, Any] = {}
    active_regions: dict[int, tuple[int, int]] = {}
    for ep in ep_frame_map:
        vp = vu.try_resolve_allintra_video_path(
            video_paths[ep], data_root=args.data_root,
            allintra_root=args.allintra_root,
            suffix=args.allintra_suffix)
        if vp is None:
            continue
        try:
            vr = vu.open_video_reader(vp)
            vrs[ep] = vr
            first_bgr = vu.read_frame_bgr(vr, 0)
            K_use, dist_use, info = vu.build_intrinsics_and_frame_mapper(
                K_raw, dist_raw, calib_w, calib_h,
                first_bgr.shape[1], first_bgr.shape[0], first_bgr)
            x0 = int(info["crop_xywh_on_video"][0])
            x1 = int(info["crop_xywh_on_video"][0] + info["crop_xywh_on_video"][2])
            active_regions[ep] = (x0, x1)
        except Exception:
            continue

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = None
    renderer_size = None
    pbar = tqdm(total=n_samples, desc="Rendering", unit="frame")

    for ep in sorted(vrs.keys()):
        vr = vrs[ep]
        x0, x1 = active_regions[ep]

        for global_i in sorted(ep_frame_map[ep]):
            device_frame_idx = int(frame_idx_mm[global_i])
            video_frame_idx = vu.clamp_frame_idx(vr, device_frame_idx)
            try:
                frame_bgr = vu.read_frame_bgr(vr, video_frame_idx)
            except Exception:
                continue
            video_h, video_w = frame_bgr.shape[:2]

            t12 = np.asarray(cam_transform[global_i], dtype=np.float64)
            T_W_C = vu.t12_to_matrix(t12)

            hand_world_verts: dict[str, np.ndarray] = {}
            fk_world_verts: dict[str, np.ndarray] = {}
            fk_faces: dict[str, np.ndarray] = {}
            for hand in ("left", "right"):
                mano_pose = np.asarray(hand_data[hand]["pose"][global_i],
                                       dtype=np.float64)
                beta_idx = int(beta_idx_arr[ep])
                beta = np.asarray(hand_data[hand]["beta"][beta_idx],
                                  dtype=np.float64)
                verts_local, _ = decoder.decode(mano_pose, beta, hand)
                t12_world = np.asarray(hand_data[hand]["world"][global_i],
                                       dtype=np.float64)
                R_world = t12_world[:9].reshape(3, 3)
                t_world = t12_world[9:12]
                hand_world_verts[hand] = verts_local @ R_world.T + t_world

                # FK mesh: always skin a right hand, then x-flip for left
                # (same convention as the original mesh script); FK faces
                # need a winding flip to match the MANO/trimesh convention.
                ja = np.asarray(hand_data[hand]["joint_angles"][global_i],
                                dtype=np.float32)
                if np.isfinite(ja).all() and np.abs(ja).sum() > 0:
                    try:
                        fk_v_local, fk_f = vu.skin_mesh_from_angles(
                            joint_angles=ja[:20], flip=False)
                        fk_v_local = fk_v_local.copy()
                        if hand == "right":
                            fk_v_local[:, 0] *= -1.0
                            fk_f = fk_f[:, [0, 2, 1]].copy()
                        fk_v_local = vu.rescale_mesh_span(fk_v_local, 0.09)
                        fk_world_verts[hand] = fk_v_local @ R_world.T + t_world
                        fk_faces[hand] = fk_f
                    except Exception:
                        pass

            frame_dir = out_dir / f"frame_{global_i:08d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for hand in ("left", "right"):
                if hand in hand_world_verts:
                    vu.save_mesh_glb(
                        hand_world_verts[hand], hand_faces[hand],
                        vu.HAND_COLORS_RGB[hand],
                        str(frame_dir / f"mano_{hand}.glb"))
                if hand in fk_world_verts:
                    vu.save_mesh_glb(
                        fk_world_verts[hand], fk_faces[hand],
                        vu.HAND_COLORS_RGB[hand],
                        str(frame_dir / f"fk_{hand}.glb"))

            # ── Self-occlusion analysis ─────────────────────────────
            K_vid = vu.intrinsics_info_to_video_K(info, K_raw)
            T_C_W = np.linalg.inv(T_W_C)
            R_C_W = T_C_W[:3, :3].astype(np.float64)
            t_C_W = T_C_W[:3, 3].astype(np.float64)

            occlusion_results: dict[str, dict] = {}
            for hand in ("left", "right"):
                if hand not in hand_world_verts:
                    continue
                from egoemg.occlusion import compute_self_occlusion
                verts_w = hand_world_verts[hand].astype(np.float64)
                verts_cam = (R_C_W @ verts_w.T).T + t_C_W
                occlusion_results[hand] = compute_self_occlusion(
                    verts_cam, hand_faces[hand], K_vid, video_h, video_w,
                    depth_eps=0.005, window_half=2)

            occ_json: dict[str, dict] = {}
            for hand, r in occlusion_results.items():
                occ_json[hand] = {
                    "occlusion_score": round(float(r["occlusion_score"]), 6),
                    "visible_ratio": round(float(r["visible_ratio"]), 6),
                    "n_visible": int(r["visible"].sum()),
                    "n_total": int(len(r["visible"])),
                    "area_weight_total": round(float(r["area_weights"].sum()), 6),
                }
            (frame_dir / "occlusion.json").write_text(
                json.dumps(occ_json, indent=2))

            occ_vis = frame_bgr.copy()
            for hand, r in occlusion_results.items():
                for i in range(len(r["visible"])):
                    u, v = int(r["u_proj"][i]), int(r["v_proj"][i])
                    if 0 <= u < video_w and 0 <= v < video_h:
                        color = (0, 255, 0) if r["visible"][i] else (0, 0, 255)
                        cv2.circle(occ_vis, (u, v), 2, color, -1,
                                   lineType=cv2.LINE_AA)
            cv2.imwrite(str(frame_dir / "occlusion_vis.png"), occ_vis)

            markers_bgr = frame_bgr.copy()
            for hand in ("left", "right"):
                kp_world = np.asarray(hand_data[hand]["keypoints"][global_i],
                                      dtype=np.float64)
                kp_valid_raw = np.asarray(
                    hand_data[hand]["keypoints_valid"][global_i], dtype=bool)
                if not kp_valid_raw.any():
                    continue
                kp_px, depth_valid = vu.project_and_map(
                    kp_world, T_W_C, K_raw, dist_raw, info)
                in_image = (
                    (kp_px[:, 0] >= 0) & (kp_px[:, 0] < video_w)
                    & (kp_px[:, 1] >= 0) & (kp_px[:, 1] < video_h))
                valid = (depth_valid & in_image & kp_valid_raw
                         & np.isfinite(kp_world).all(axis=1))
                if valid.sum() > 0:
                    markers_bgr = vu.draw_skeleton(
                        markers_bgr, kp_px, valid,
                        vu.HAND_COLORS_BGR[hand], label=hand[0].upper())
            cv2.imwrite(str(frame_dir / "markers.png"), markers_bgr)

            if args.render_mode == "mesh":
                frame_undist = cv2.undistort(frame_bgr, K_vid, dist_raw,
                                             None, K_vid)
                if (video_w, video_h) != renderer_size:
                    if renderer is not None:
                        renderer.delete()
                    import pyrender
                    renderer = pyrender.OffscreenRenderer(video_w, video_h)
                    renderer_size = (video_w, video_h)
                meshes = [
                    (hand_world_verts[h], hand_faces[h], vu.HAND_COLORS_RGB[h])
                    for h in ("left", "right")
                ]
                frame_bgr = _render_mesh_overlay(
                    frame_undist, meshes, T_W_C, K_vid,
                    renderer, args.mesh_alpha)
            else:
                for hand in ("left", "right"):
                    if hand not in hand_world_verts:
                        continue
                    verts_px, depth_valid = vu.project_and_map(
                        hand_world_verts[hand], T_W_C, K_raw, dist_raw, info)
                    in_image = (
                        (verts_px[:, 0] >= 0) & (verts_px[:, 0] < video_w)
                        & (verts_px[:, 1] >= 0) & (verts_px[:, 1] < video_h))
                    valid = depth_valid & in_image
                    vu.draw_wireframe(
                        frame_bgr, verts_px, valid, hand_faces[hand],
                        vu.HAND_COLORS_BGR[hand], args.line_width)

            frame_bgr = vu.draw_text_block(
                frame_bgr,
                [f"global={global_i} ep={ep}", "R: orange  L: blue"])
            cv2.imwrite(str(frame_dir / "rendered.png"), frame_bgr)
            pbar.update(1)

    pbar.close()
    if renderer is not None:
        renderer.delete()
    print(f"Done. Saved {n_samples} frames to {out_dir.resolve()}")
    return 0


# ── markers mode ────────────────────────────────────────────────────────────

def run_markers(args: argparse.Namespace) -> int:
    from tqdm import tqdm

    manifest = vu.load_manifest(args.memmap_dir)
    md = vu.load_metadata(args.memmap_dir)
    ep_ids = vu.decode_bytes(md["episode_id"])
    starts = md["episode_start_idx"].astype(np.int64)
    ends = md["episode_end_idx"].astype(np.int64)

    ep_idx = None
    for i, eid in enumerate(ep_ids):
        if eid == args.episode_id:
            ep_idx = i
            break
    if ep_idx is None:
        raise ValueError(f"Unknown episode id: {args.episode_id}")
    start = int(starts[ep_idx])
    length = int(ends[ep_idx]) - start

    calib = vu.load_calibration(vu.resolve_calibration_path(
        args.data_root, args.calibration_json))
    K, dist, calib_w, calib_h = calib.K, calib.dist, calib.width, calib.height

    cam_tf_mm = vu.load_memmap(args.memmap_dir, manifest, "mocap_head_transform")
    kp_left_mm = vu.load_memmap(args.memmap_dir, manifest, "mocap_left_keypoints")
    valid_left_mm = vu.load_memmap(args.memmap_dir, manifest, "mocap_left_valid")
    kp_right_mm = vu.load_memmap(args.memmap_dir, manifest, "mocap_right_keypoints")
    valid_right_mm = vu.load_memmap(args.memmap_dir, manifest, "mocap_right_valid")
    frame_idx_mm = vu.load_memmap(args.memmap_dir, manifest, "image_head_frame_index")

    allintra_path = vu.resolve_allintra_video_path(
        vu.decode_bytes(md["episode_head_video_path"])[ep_idx],
        data_root=args.data_root, allintra_root=args.allintra_root,
        suffix=args.allintra_suffix)

    import cv2
    cap = cv2.VideoCapture(str(allintra_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_frame_indices = np.asarray(
        frame_idx_mm[start:start + length], dtype=np.int64)
    seen: dict[int, int] = {}
    unique_frames = []
    for offset, vfi in enumerate(raw_frame_indices):
        if vfi >= 0 and vfi not in seen:
            seen[vfi] = offset
            unique_frames.append((int(vfi), offset))

    total_video_frames = len(unique_frames)
    print(f"[{args.episode_id}] {total_video_frames} unique video frames "
          f"in episode ({length} memmap frames)")
    strided_frames = unique_frames[0::args.stride]
    if args.max_frames > 0:
        strided_frames = strided_frames[:args.max_frames]

    output = Path(args.output) if args.output else (
        Path(args.output_dir) / f"{args.episode_id}_markers.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output), fourcc,
                             int(round(fps / args.stride)), (video_w, video_h))

    intrinsics_info = None
    pbar = tqdm(strided_frames, desc=f"Processing {args.episode_id}",
                unit="vframe")
    for video_frame_idx, memmap_offset in pbar:
        global_i = start + memmap_offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        if intrinsics_info is None:
            K_use, dist_use, intrinsics_info = \
                vu.build_intrinsics_and_frame_mapper(
                    K, dist, calib_w, calib_h, video_w, video_h, frame)
        T_W_C = vu.t12_to_matrix(np.asarray(cam_tf_mm[global_i]))
        for hand, kp_mm, valid_mm, color in (
            ("left", kp_left_mm, valid_left_mm, (0, 0, 255)),
            ("right", kp_right_mm, valid_right_mm, (0, 255, 0)),
        ):
            kp = np.asarray(kp_mm[global_i], dtype=np.float64)
            valid = np.asarray(valid_mm[global_i], dtype=bool)
            if not valid.any():
                continue
            pts_raw, depth = vu.project_and_map(
                kp[valid], T_W_C, K_use, dist_use, intrinsics_info)
            valid_mask = depth & (pts_raw[:, 0] >= 0) & (pts_raw[:, 0] < video_w) \
                & (pts_raw[:, 1] >= 0) & (pts_raw[:, 1] < video_h)
            vu.draw_skeleton(frame, pts_raw, valid_mask, color,
                             edges=vu.HAND_BONES)
        frame = vu.draw_text_block(frame, [
            f"{args.episode_id}  frame={video_frame_idx}",
            "L=red  R=green",
        ], line_height=25)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\nSaved: {output}")
    return 0


# ── crops mode ──────────────────────────────────────────────────────────────

def run_crops(args: argparse.Namespace) -> int:
    import cv2

    crops_dir = Path(args.crops_dir)
    manifest_path = crops_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        ep_ids = manifest["episode_ids"]
        patch_size = manifest.get("patch_size", 256)
        print(f"Found {len(ep_ids)} episodes, patch_size={patch_size}")
    else:
        ep_ids = sorted(
            p.name[:-5] for p in crops_dir.glob("episode_*.lmdb"))
        patch_size = 256
        print(f"Manifest missing; found {len(ep_ids)} LMDB episodes")

    if args.episodes is not None:
        indices = [int(x) for x in args.episodes.split(",")]
    else:
        indices = list(range(len(ep_ids)))

    for ep_i in indices:
        if ep_i >= len(ep_ids):
            print(f"  Episode index {ep_i} out of range, skipping")
            continue
        ep_id = ep_ids[ep_i]
        lmdb_path = crops_dir / f"{ep_id}.lmdb"
        if not lmdb_path.exists():
            print(f"  [{ep_id}] LMDB not found, skipping")
            continue

        all_keys = vu.list_lmdb_keys(lmdb_path)
        if not all_keys:
            print(f"  [{ep_id}] empty LMDB, skipping")
            continue

        vfis = sorted(set(int(k.split("_")[0]) for k in all_keys))
        print(f"  [{ep_id}] {len(all_keys)} crops, {len(vfis)} frames")
        n_sample = min(args.num_frames, len(vfis))
        step = max(len(vfis) // n_sample, 1)
        sampled_vfis = vfis[::step][:n_sample]

        rows = []
        for vfi in sampled_vfis:
            left = vu.read_crop_from_lmdb(lmdb_path, f"{vfi:08d}_L")
            right = vu.read_crop_from_lmdb(lmdb_path, f"{vfi:08d}_R")
            left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR) if left is not None \
                else np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR) if right is not None \
                else np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            if not np.any(left):
                cv2.putText(left, "NO L", (10, patch_size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not np.any(right):
                cv2.putText(right, "NO R", (10, patch_size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            left_lab = left.copy()
            right_lab = right.copy()
            cv2.putText(left_lab, f"L vfi={vfi}", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(right_lab, f"R vfi={vfi}", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            rows.append(np.concatenate([left_lab, right_lab], axis=1))

        cols = args.grid_cols
        n_rows = (len(rows) + cols - 1) // cols
        grid_rows = []
        for r in range(n_rows):
            chunks = []
            for c in range(cols):
                idx = r * cols + c
                chunks.append(rows[idx] if idx < len(rows)
                              else np.zeros_like(rows[0]))
            grid_rows.append(np.concatenate(chunks, axis=1))
        grid = np.concatenate(grid_rows, axis=0)
        h, w = rows[0].shape[:2]
        for r in range(1, n_rows):
            y = r * h
            if y < grid.shape[0]:
                grid[y, :] = 30
        for c in range(1, cols):
            x = c * w
            if x < grid.shape[1]:
                grid[:, x] = 30

        out_path = Path(args.output_dir) / f"{ep_id}.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"  [{ep_id}] saved {out_path} ({grid.shape[1]}x{grid.shape[0]})")

    print(f"\nDone. Output in {args.output_dir}")
    return 0


# ── fk_vs_mano mode ─────────────────────────────────────────────────────────

def run_fk_vs_mano(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image

    out_dir = Path(args.output_dir)
    gbl_dir = out_dir / "glb"
    crop_dir = out_dir / "crops"
    gbl_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["emg", "joint_angles", "mano", "labels"]
    if args.add_video_fields:
        modalities.append("video_index")
    ds = vu.make_memmap_dataset(
        memmap_dir=args.memmap_dir, hand=args.hand,
        window_length=args.window_length, stride=args.stride,
        modalities=modalities,
        mano_npy_dir=args.mano_npy_dir,
        emg_layout="emg2pose_interpolate16",
        emg2pose_channel_indices=[10, 12, 0, 1, 2, 4, 5, 6],
        channel_interpolate=False,
        norm_mode="per-dataset",
        norm_stats_path="./assets/per_dataset_norm_stats.json",
        jitter=False,
    )
    print(f"  {len(ds)} windows, {len(ds._episode_id)} episodes")

    decoder = vu.ManoMeshDecoder(args.mano_model_path, args.device)
    print("  MANO decoder ready")

    rng = np.random.RandomState(args.seed)
    if args.episode is not None:
        ep_mask = np.asarray(ds._block_episode_idx) == args.episode
        valid_indices = np.where(ep_mask)[0]
    else:
        valid_indices = np.arange(len(ds))
    if len(valid_indices) == 0:
        print("No windows found!")
        return 0
    n = min(args.num_samples, len(valid_indices))
    chosen = rng.choice(valid_indices, size=n, replace=False)
    hand_code = "L" if args.hand == "left" else "R"

    torch.set_grad_enabled(False)
    for i, idx in enumerate(chosen):
        sample = ds[int(idx)]
        ep_id = sample.get("episode_id", "unknown")
        center = args.window_length // 2
        if "mano_pose" not in sample or "mano_beta" not in sample:
            print(f"  [{i+1}/{n}] idx={idx}: no MANO params, skipping")
            continue
        mano_pose_mid = np.array(sample["mano_pose"][center], copy=True)
        mano_beta = np.array(sample["mano_beta"], copy=True)
        ja_mid = np.array(sample["joint_angles"][:, center], copy=True)

        fk_verts, fk_faces = vu.skin_mesh_from_angles(
            joint_angles=ja_mid[:20], flip=(args.hand == "left"))
        fk_verts = fk_verts.copy()
        mano_verts, mano_faces = decoder.decode(mano_pose_mid, mano_beta)
        fk_span = fk_verts.max(axis=0) - fk_verts.min(axis=0)
        mano_span = mano_verts.max(axis=0) - mano_verts.min(axis=0)
        if np.median(fk_span) > 1e-6:
            scale_factor = float(np.median(mano_span) / np.median(fk_span))
            fk_verts = fk_verts * scale_factor

        win_start = int(np.asarray(sample["window_start_idx"]))
        gbl_path = gbl_dir / f"{ep_id}_{args.hand}_win{win_start:08d}.glb"
        vu.save_comparison_glb(gbl_path, fk_verts, fk_faces,
                               mano_verts, mano_faces)
        print(f"  [{i+1}/{n}] {gbl_path.name}")

        if args.per_episode_crops_dir is not None:
            if "image_head_frame_index" in sample:
                vid_frame = int(np.asarray(
                    sample["image_head_frame_index"])[center])
            else:
                vid_frame = win_start
            crop_img = vu.load_episode_crop(
                args.per_episode_crops_dir, ep_id, vid_frame, hand_code)
            if crop_img is not None:
                crop_path = crop_dir / f"{ep_id}_{args.hand}_win{win_start:08d}.png"
                Image.fromarray(crop_img).save(str(crop_path))
                print(f"         crop: {crop_path.name}")
            else:
                print(f"         crop: not found for frame {vid_frame}")

    print(f"\nDone. GLB files: {gbl_dir}")
    if args.per_episode_crops_dir is not None:
        print(f"Crop images: {crop_dir}")
    return 0


# ── align mode ──────────────────────────────────────────────────────────────

def run_align(args: argparse.Namespace) -> int:
    import cv2
    from egoemg.video_io import resolve_allintra_video_path

    md = vu.load_metadata(args.memmap_dir)
    manifest = vu.load_manifest(args.memmap_dir)
    fi = vu.load_memmap(args.memmap_dir, manifest, "image_head_frame_index")

    def mm(name: str) -> np.memmap:
        return vu.load_memmap(args.memmap_dir, manifest, name)

    ep_idx = args.session_index
    s, e = int(md["episode_start_idx"][ep_idx]), int(md["episode_end_idx"][ep_idx])
    ep_id = vu.decode_bytes(np.asarray([md["episode_id"][ep_idx]]))[0]
    raw_video = vu.decode_bytes(np.asarray([md["episode_head_video_path"][ep_idx]]))[0]
    video_path = resolve_allintra_video_path(
        raw_video_path=raw_video, data_root=args.data_root,
        allintra_root=args.allintra_root)
    print(f"episode {ep_idx} ({ep_id}): rows [{s}, {e}), video {video_path}")

    calib = vu.load_calibration(args.calibration_path)
    vr = vu.open_video_reader(video_path)
    n_video_frames = len(vr)
    frame0 = vr[0].asnumpy()
    K, dist, info = vu.build_intrinsics_and_frame_mapper(
        calib.K, calib.dist, calib.width, calib.height,
        frame0.shape[1], frame0.shape[0], frame0)
    video_h, video_w = frame0.shape[:2]
    print(f"video: {n_video_frames} frames, {video_w}x{video_h}")

    transforms = mm("mocap_head_transform")
    kp_l = mm("mocap_left_keypoints")
    kp_r = mm("mocap_right_keypoints")
    valid_l = mm("mocap_left_valid")
    valid_r = mm("mocap_right_valid")
    lv = mm("generated_label_valid")
    stale = mm("image_head_stale")

    rows = np.linspace(s, e - 1, args.samples_per_action * 40).astype(np.int64)
    rows = [r for r in rows if not bool(stale[r]) and bool(lv[r].all())][
        : args.samples_per_action * 8
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        frame_idx = int(fi[r])
        if frame_idx < 0 or frame_idx >= n_video_frames:
            print(f"  row {r}: frame index {frame_idx} out of video range, skip")
            continue
        frame = vu.read_frame_bgr(vr, frame_idx)
        T_W_C = vu.t12_to_matrix(np.asarray(transforms[r]))
        for hand_name, kp_mm, valid_mm, color in (
            ("left", kp_l, valid_l, (0, 255, 0)),
            ("right", kp_r, valid_r, (0, 0, 255)),
        ):
            marker_world = np.asarray(kp_mm[r], dtype=np.float64)
            marker_valid = np.asarray(valid_mm[r], dtype=bool)
            raw, depth_valid = vu.project_and_map(
                marker_world, T_W_C, K, dist, info)
            in_img = (
                (raw[:, 0] >= 0) & (raw[:, 0] < video_w)
                & (raw[:, 1] >= 0) & (raw[:, 1] < video_h))
            good = marker_valid & depth_valid & in_img
            for x, y, ok in zip(raw[:, 0], raw[:, 1], good):
                if ok:
                    cv2.circle(frame, (int(x), int(y)), 6, color, -1)
        frame = vu.draw_text_block(
            frame, [f"row={r} video_frame={frame_idx}"], line_height=40)
        out = out_dir / f"row_{r}_vf_{frame_idx}.png"
        cv2.imwrite(str(out), frame)
        print(f"  row {r} -> {out}")

    print(f"\nWrote {len(rows)} overlays to {out_dir}")
    return 0


# ── parser ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--memmap-dir", type=Path,
                        default=Path("data/EgoEMG_unified_memmap"))
    common.add_argument("--data-root", type=Path, default=Path("data"))
    common.add_argument("--allintra-root", type=Path,
                        default=Path("data/EgoEMG_allintra"))
    common.add_argument("--allintra-suffix", default="_allintra")
    common.add_argument("--output-dir", type=Path, default=None)
    common.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    common.add_argument("--seed", type=int, default=42)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("vision", parents=[common],
                       description="EgoEmgVisionDataset samples -> PNG")
    p.add_argument("--video-root", type=Path, default=Path("data/EgoEMG"))
    p.add_argument("--vision-index-dir", type=Path, default=None)
    p.add_argument("--auto-build-index", action="store_true")
    p.add_argument("--calibration-path", type=Path, default=None)
    p.add_argument("--target-hand", default="both",
                   choices=["left", "right", "both"])
    p.add_argument("--allowed-episode-ids", nargs="*", default=None)
    p.add_argument("--allowed-subjects", nargs="*", default=None)
    p.add_argument("--allowed-splits", nargs="*", default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--sample-indices", nargs="*", type=int, default=None)
    p.add_argument("--joint-radius", type=int, default=3)
    p.add_argument("--marker-radius", type=int, default=3)
    p.add_argument("--bbox-line-width", type=int, default=2)
    p.add_argument("--raw-only", action="store_true")
    p.add_argument("--max-panel-width", type=int, default=1280)

    p = sub.add_parser("timeline", parents=[common],
                       description="EMG / joint angles / MANO time series -> PNG")
    p.add_argument("--episode", type=int, default=3)
    p.add_argument("--hand", default="right", choices=["left", "right"])
    p.add_argument("--offset", type=int, default=100000)
    p.add_argument("--window", type=int, default=2000)
    p.add_argument("--out-path", type=Path, default=None)

    p = sub.add_parser("mano", parents=[common],
                       description="GT MANO mesh + markers -> GLB")
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--hand", default="right", choices=["left", "right"])
    p.add_argument("--offset", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--mano-model-path", type=Path, default=None)
    p.add_argument("--mano-npy-dir", type=Path, default=None)

    p = sub.add_parser("mesh", parents=[common],
                       description="MANO/FK mesh overlay on head-view frames")
    p.add_argument("--mano-model-path", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--line-width", type=int, default=1)
    p.add_argument("--render-mode", default="mesh",
                   choices=["wireframe", "mesh"])
    p.add_argument("--mesh-alpha", type=float, default=0.7)
    p.add_argument("--calibration-path", type=Path, default=None)

    p = sub.add_parser("markers", parents=[common],
                       description="Mocap marker reprojection over an episode -> MP4")
    p.add_argument("--episode-id", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--calibration-json", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)

    p = sub.add_parser("crops", parents=[common],
                       description="Pre-cropped hand patch grid -> JPG")
    p.add_argument("--crops-dir", type=Path,
                   default=Path("data/EgoEMG_v2_crops"))
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--episodes", type=str, default=None)
    p.add_argument("--grid-cols", type=int, default=4)

    p = sub.add_parser("fk_vs_mano", parents=[common],
                       description="UmeTrack FK vs MANO mesh comparison -> GLB")
    p.add_argument("--mano-npy-dir", type=Path, default=None)
    p.add_argument("--per-episode-crops-dir", type=Path, default=None)
    p.add_argument("--hand", default="right", choices=["left", "right"])
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--window-length", type=int, default=7790)
    p.add_argument("--stride", type=int, default=7790)
    p.add_argument("--add-video-fields", action="store_true")
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--mano-model-path", type=Path, default=None)

    p = sub.add_parser("align", parents=[common],
                       description="ShowEE session frame-alignment check -> PNG")
    p.add_argument("--calibration-path", type=Path, required=True)
    p.add_argument("--session-index", type=int, required=True)
    p.add_argument("--samples-per-action", type=int, default=3)

    return parser


MODES = {
    "vision": run_vision,
    "timeline": run_timeline,
    "mano": run_mano,
    "mesh": run_mesh,
    "markers": run_markers,
    "crops": run_crops,
    "fk_vs_mano": run_fk_vs_mano,
    "align": run_align,
}


def main() -> int:
    vu.setup_headless_environment()
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = Path("/tmp/egoemg_viz") / args.mode
    return MODES[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
