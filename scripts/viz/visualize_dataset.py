#!/usr/bin/env python3
"""Unified dataset-visualization entrypoint.

Replaces the per-mode dataset viz scripts with one multi-function command::

    python scripts/viz/visualize_dataset.py <mode> [mode options]

Modes (dataset itself, ground truth only):

  vision      video replay: MANO mesh projection + mocap markers +
              bbox overlaid on head-view frames -> MP4
  timeline    EMG / joint angles / MANO multi-panel time series -> PNG
  mesh        MANO/FK mesh overlay on head-view frames -> PNG + GLB +
              occlusion metrics (pyrender); --glb-only exports the
              mesh + mocap-marker GLBs without any video
  fk_vs_mano  UmeTrack FK vs MANO mesh comparison -> GLB

Shared options (available in every mode):

    --memmap-dir      data/EgoEMG_full_memmap
    --data-root       data
    --allintra-root   data/EgoEMG_videos
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

# Rendering modes import matplotlib/pyrender lazily; pin the headless
# backends before any of that runs so backend probing never hangs on a
# headless box.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emg2pose_viz_runtime" / "mpl"))

from egoemg.visualization import viz_utils as vu  # noqa: E402

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
        emg_field_preference=args.emg_preference,
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


# ── mesh mode ───────────────────────────────────────────────────────────────

FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def _render_mesh_overlay(frame_bgr: np.ndarray,
                         hand_meshes: list[tuple[np.ndarray, np.ndarray,
                                                 tuple[int, int, int]]],
                         T_W_C: np.ndarray, K_vid: np.ndarray,
                         renderer: Any, alpha: float) -> np.ndarray:
    """Render world-space hand meshes over an undistorted frame.

    The frame must be undistorted with the same K_vid (see
    ``cv2.initUndistortRectifyMap(K_vid, dist, None, K_vid)``) so the
    render aligns pixel-exactly with pinhole-projected overlays.
    Pyrender platform (egl/osmesa) follows PYOPENGL_PLATFORM.
    """
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

    cam_tracked = vu.load_memmap(args.memmap_dir, manifest, "mocap_head_valid")
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
    active_info: dict[int, dict[str, Any]] = {}
    for ep in ep_frame_map:
        if args.glb_only:
            break  # GLB-only: world-space meshes need no videos
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
            _, _, info = vu.build_intrinsics_and_frame_mapper(
                K_raw, dist_raw, calib_w, calib_h,
                first_bgr.shape[1], first_bgr.shape[0], first_bgr)
            active_info[ep] = info
        except Exception:
            continue

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = None
    renderer_size = None
    pbar = tqdm(total=n_samples, desc="Rendering", unit="frame")

    for ep in sorted(ep_frame_map.keys()):
        vr = vrs.get(ep)
        info = active_info.get(ep)

        for global_i in sorted(ep_frame_map[ep]):
            if not args.glb_only:
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
            pred_markers: dict[str, np.ndarray] = {}
            fk_world_verts: dict[str, np.ndarray] = {}
            fk_faces: dict[str, np.ndarray] = {}
            for hand in ("left", "right"):
                mano_pose = np.asarray(hand_data[hand]["pose"][global_i],
                                       dtype=np.float64)
                beta_idx = int(beta_idx_arr[ep])
                beta = np.asarray(hand_data[hand]["beta"][beta_idx],
                                  dtype=np.float64)
                verts_local, _ = decoder.decode(mano_pose, beta, hand)
                R_world, t_world = vu.t12_world_rt(
                    hand_data[hand]["world"][global_i])
                hand_world_verts[hand] = vu.verts_world_from_local(
                    verts_local, R_world, t_world)
                pred_markers[hand] = vu.verts_world_from_local(
                    decoder.marker_vertices(verts_local), R_world, t_world)

                # FK mesh from UmeTrack joint angles; skin/mirror/winding
                # convention lives in vu.fk_mesh_world.
                ja = np.asarray(hand_data[hand]["joint_angles"][global_i],
                                dtype=np.float32)
                fk = vu.fk_mesh_world(ja, R_world, t_world,
                                      mirror_x=(hand == "right"),
                                      anchor_verts=verts_local)
                if fk is not None:
                    fk_world_verts[hand], fk_faces[hand] = fk

            frame_dir = out_dir / f"frame_{global_i:08d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for hand in ("left", "right"):
                if hand in hand_world_verts:
                    kp = np.asarray(hand_data[hand]["keypoints"][global_i],
                                    dtype=np.float64)
                    kp_valid = np.asarray(
                        hand_data[hand]["keypoints_valid"][global_i], dtype=bool)
                    gt = kp[kp_valid & np.isfinite(kp).all(axis=1)]
                    vu.save_glb_with_markers(
                        frame_dir / f"mano_{hand}.glb",
                        hand_world_verts[hand], hand_faces[hand],
                        mesh_color=vu.HAND_COLORS_RGB[hand] + (255,),
                        gt_markers=gt if len(gt) else None,
                        pred_markers=pred_markers[hand])
                if hand in fk_world_verts:
                    vu.save_mesh_glb(
                        fk_world_verts[hand], fk_faces[hand],
                        vu.HAND_COLORS_RGB[hand],
                        str(frame_dir / f"fk_{hand}.glb"))

            if not args.glb_only:
                # ── Self-occlusion analysis ────────────────────────
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
                        "occlusion_score":
                            round(float(r["occlusion_score"]), 6),
                        "visible_ratio": round(float(r["visible_ratio"]), 6),
                        "n_visible": int(r["visible"].sum()),
                        "n_total": int(len(r["visible"])),
                        "area_weight_total":
                            round(float(r["area_weights"].sum()), 6),
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
                    markers_bgr = vu.project_draw_keypoints(
                        markers_bgr,
                        hand_data[hand]["keypoints"][global_i],
                        hand_data[hand]["keypoints_valid"][global_i],
                        T_W_C, K_raw, dist_raw, info,
                        vu.HAND_COLORS_BGR[hand], label=hand[0].upper())
                cv2.imwrite(str(frame_dir / "markers.png"), markers_bgr)

                if args.render_mode == "mesh":
                    K_vid = vu.intrinsics_info_to_video_K(info, K_raw)
                    if (video_w, video_h) != renderer_size:
                        if renderer is not None:
                            renderer.delete()
                        renderer = vu.make_pyrender_renderer(video_w, video_h)
                        renderer_size = (video_w, video_h)
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K_vid, dist_raw, None, K_vid, (video_w, video_h),
                        cv2.CV_32FC1)
                    frame_undist = cv2.remap(
                        frame_bgr, mapx, mapy, cv2.INTER_LINEAR)
                    meshes = [
                        (hand_world_verts[h], hand_faces[h],
                         vu.HAND_COLORS_RGB[h])
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
                            hand_world_verts[hand], T_W_C, K_raw, dist_raw,
                            info)
                        in_image = (
                            (verts_px[:, 0] >= 0)
                            & (verts_px[:, 0] < video_w)
                            & (verts_px[:, 1] >= 0)
                            & (verts_px[:, 1] < video_h))
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


# ── vision video mode ───────────────────────────────────────────────────────

def _mesh_projected_bbox(verts_px: np.ndarray, valid: np.ndarray,
                         img_w: int, img_h: int, pad: int) -> np.ndarray | None:
    """Tight in-image bbox around projected mesh vertices (None if none)."""
    px = verts_px[valid]
    if len(px) == 0:
        return None
    x0 = max(0, int(np.floor(px[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(px[:, 1].min())) - pad)
    x1 = min(img_w - 1, int(np.ceil(px[:, 0].max())) + pad)
    y1 = min(img_h - 1, int(np.ceil(px[:, 1].max())) + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _precrop_affine(
    markers_world: np.ndarray,
    markers_valid: np.ndarray,
    t_w_c: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    intrinsics_info: dict[str, Any],
    video_w: int,
    hand: str,
    crop_size: int,
) -> np.ndarray | None:
    """Recreate the affine transform used to make a stored hand crop."""
    from egoemg.datasets.egoemg_vision_dataset import (
        _expand_to_aspect_ratio,
        _gen_trans_from_patch_cv,
        _get_bbox,
    )

    marker_px, depth_valid = vu.project_and_map(
        markers_world, t_w_c, k, dist, intrinsics_info)
    marker_px = marker_px.astype(np.float32)
    valid = np.asarray(markers_valid, dtype=bool) & depth_valid
    if hand == "left":
        marker_px[:, 0] = (video_w - 1) - marker_px[:, 0]
    if valid.sum() < 2:
        return None
    keypoints = np.concatenate(
        [marker_px, valid.astype(np.float32)[:, None]], axis=1)
    center, scale = _get_bbox(keypoints, rescale=1.2)
    scale = _expand_to_aspect_ratio(scale, (192, 256))
    bbox_size = float(max(scale[0], scale[1]))
    return _gen_trans_from_patch_cv(
        float(center[0]), float(center[1]), bbox_size, bbox_size,
        crop_size, crop_size, scale=1.0, rot=0.0)


def run_vision_video(args: argparse.Namespace) -> int:
    """Video replay of one episode: head-view frames with MANO/FK mesh
    projection, mocap marker skeletons and per-hand bboxes overlaid."""
    import io
    from PIL import Image
    from tqdm import tqdm
    import cv2

    manifest = vu.load_manifest(args.memmap_dir)
    md = vu.load_metadata(args.memmap_dir)
    ep_ids = vu.decode_bytes(md["episode_id"])
    starts = md["episode_start_idx"].astype(np.int64)
    ends = md["episode_end_idx"].astype(np.int64)
    beta_idx_arr = md["episode_beta_idx"]

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
    frame_idx_mm = vu.load_memmap(args.memmap_dir, manifest,
                                  "image_head_frame_index")
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
            "keypoints": vu.load_memmap(args.memmap_dir, manifest,
                                        f"mocap_{hand}_keypoints"),
            "keypoints_valid": vu.load_memmap(args.memmap_dir, manifest,
                                              f"mocap_{hand}_valid"),
        }

    decoder = vu.ManoMeshDecoder(args.mano_model_path, args.device)
    faces_right = decoder._faces
    faces_left = faces_right[:, [0, 2, 1]]
    hand_faces = {"right": faces_right, "left": faces_left}

    allintra_path = vu.resolve_allintra_video_path(
        vu.decode_bytes(md["episode_head_video_path"])[ep_idx],
        data_root=args.data_root, allintra_root=args.allintra_root,
        suffix=args.allintra_suffix)
    vr = vu.open_video_reader(allintra_path)
    fps = vr.get_avg_fps()
    video_h, video_w = vu.read_frame_bgr(vr, 0).shape[:2]

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

    # Crop production only writes frames with >=2 in-view valid markers, so a
    # few episode-head frames (notably frame 0 everywhere) have no crop keys
    # at all. Drop them from the selection with a notice instead of failing
    # the whole render; genuinely missing crops for retained frames are still
    # a hard error below.
    crop_lmdb_probe = Path(args.crops_dir) / f"{args.episode_id}.lmdb"
    if crop_lmdb_probe.is_dir():
        import lmdb as _lmdb_probe

        _env = _lmdb_probe.open(str(crop_lmdb_probe), readonly=True, lock=False,
                                readahead=False)
        with _env.begin() as _txn:
            def _has_crop(vfi: int) -> bool:
                return (
                    _txn.get(f"{vfi:08d}_L".encode()) is not None
                    and _txn.get(f"{vfi:08d}_R".encode()) is not None
                )
            _kept = [(vfi, mfi) for vfi, mfi in strided_frames if _has_crop(vfi)]
        _env.close()
        _dropped = len(strided_frames) - len(_kept)
        if _dropped:
            print(f"[{args.episode_id}] skipping {_dropped} selected frame(s) "
                  "without precomputed crops (no valid markers at capture time)")
        strided_frames = _kept
        if not strided_frames:
            raise RuntimeError(
                f"None of the selected frames have precomputed crops in "
                f"{crop_lmdb_probe}; choose different frames/stride.")

    output = Path(args.output) if args.output else (
        Path(args.output_dir) / f"{args.episode_id}_vision.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Output fps = real-time (fps/stride) floored at 15: low strides play at
    # native speed, high strides fast-forward smoothly instead of becoming a
    # 1 fps slideshow or a 60x flash.
    out_fps = max(15.0, fps / args.stride)
    crop_size = 256
    manifest_path = Path(args.crops_dir) / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            crop_size = int(json.load(f).get("patch_size", crop_size))
    crop_lmdb = Path(args.crops_dir) / f"{args.episode_id}.lmdb"
    if not crop_lmdb.is_dir():
        raise FileNotFoundError(
            f"Precomputed crop LMDB is required for vision output: {crop_lmdb}. "
            "Create/download the episode crops first; vision never crops from "
            "the overlay bbox at runtime.")

    import lmdb
    crop_env = lmdb.open(str(crop_lmdb), readonly=True, lock=False,
                         readahead=False)
    crop_txn = crop_env.begin()
    missing_crop_keys = [
        f"{video_frame_idx:08d}_{hand_code}"
        for video_frame_idx, _ in strided_frames
        for hand_code in ("L", "R")
        if crop_txn.get(f"{video_frame_idx:08d}_{hand_code}".encode()) is None
    ]
    if missing_crop_keys:
        crop_env.close()
        preview = ", ".join(missing_crop_keys[:3])
        raise RuntimeError(
            f"{len(missing_crop_keys)} required precomputed crops are missing "
            f"from {crop_lmdb} (for example: {preview}). Refusing to create "
            "a misleading partial/black crop video.")
    # All three writers are created only after every selected frame's crop
    # key has been verified, so a validation failure leaves no partial
    # output files behind.
    writer = vu.open_mp4_writer(output, out_fps, (video_w, video_h))
    crop_writers = {
        hand: vu.open_mp4_writer(
            output.parent / f"{args.episode_id}_{hand}_crop.mp4",
            out_fps, (crop_size, crop_size))
        for hand in ("left", "right")
    }

    def read_precrop(video_frame_idx: int, hand: str) -> np.ndarray | None:
        if crop_txn is None:
            return None
        key = f"{video_frame_idx:08d}_{'L' if hand == 'left' else 'R'}"
        encoded = crop_txn.get(key.encode())
        if encoded is None:
            return None
        return cv2.cvtColor(
            np.asarray(Image.open(io.BytesIO(encoded))), cv2.COLOR_RGB2BGR)

    intrinsics_info = None
    renderer = None
    renderer_size = None
    pbar = tqdm(strided_frames, desc=f"Overlay {args.episode_id}",
                unit="vframe")
    for video_frame_idx, memmap_offset in pbar:
        global_i = start + memmap_offset
        frame = vu.read_frame_bgr(vr, vu.clamp_frame_idx(vr, video_frame_idx))
        if intrinsics_info is None:
            K_use, dist_use, intrinsics_info = \
                vu.build_intrinsics_and_frame_mapper(
                    K, dist, calib_w, calib_h, video_w, video_h, frame)
            K_vid = vu.intrinsics_info_to_video_K(intrinsics_info, K)
            # Undistorted basis: mesh (pyrender), markers, bbox all project
            # through the same ideal pinhole, so they align pixel-exactly.
            mapx, mapy = cv2.initUndistortRectifyMap(
                K_vid, dist, None, K_vid, (video_w, video_h), cv2.CV_32FC1)
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        T_W_C = vu.t12_to_matrix(np.asarray(cam_tf_mm[global_i]))
        beta_idx = int(beta_idx_arr[ep_idx])
        hand_verts: dict[str, np.ndarray] = {}
        for hand in ("left", "right"):
            pose = np.asarray(hand_data[hand]["pose"][global_i],
                              dtype=np.float64)
            t12_world = np.asarray(hand_data[hand]["world"][global_i],
                                   dtype=np.float64)
            if not (np.isfinite(pose).all() and np.isfinite(t12_world).all()
                    and np.abs(t12_world).sum() > 1e-6):
                continue  # no valid MANO supervision for this row
            beta = np.asarray(hand_data[hand]["beta"][beta_idx],
                              dtype=np.float64)
            R_w, t_w = vu.t12_world_rt(t12_world)
            verts_local, _ = decoder.decode(pose, beta, hand)
            hand_verts[hand] = vu.verts_world_from_local(verts_local, R_w, t_w)

        if args.render_mode == "mesh":
            if (video_w, video_h) != renderer_size:
                if renderer is not None:
                    renderer.delete()
                renderer = vu.make_pyrender_renderer(video_w, video_h)
                renderer_size = (video_w, video_h)
            meshes = [
                (hand_verts[h], hand_faces[h], vu.HAND_COLORS_RGB[h])
                for h in ("left", "right") if h in hand_verts
            ]
            frame = _render_mesh_overlay(frame, meshes, T_W_C, K_vid,
                                         renderer, args.mesh_alpha)

        for hand, verts_w in hand_verts.items():
            verts_px, depth_valid = vu.project_pinhole(verts_w, T_W_C, K_vid)
            in_image = (
                (verts_px[:, 0] >= 0) & (verts_px[:, 0] < video_w)
                & (verts_px[:, 1] >= 0) & (verts_px[:, 1] < video_h))
            valid = depth_valid & in_image
            if args.render_mode == "wireframe":
                vu.draw_wireframe(frame, verts_px, valid, hand_faces[hand],
                                  vu.HAND_COLORS_BGR[hand], args.line_width)
            bbox = _mesh_projected_bbox(
                verts_px, valid, video_w, video_h, args.bbox_pad)
            if bbox is not None:
                frame = vu.draw_bbox(
                    frame, bbox, vu.HAND_COLORS_BGR[hand], 2)

        for hand in ("left", "right"):
            frame = vu.project_draw_keypoints_pinhole(
                frame, hand_data[hand]["keypoints"][global_i],
                hand_data[hand]["keypoints_valid"][global_i],
                T_W_C, K_vid, vu.HAND_COLORS_BGR[hand],
                label=hand[0].upper())

        frame = vu.draw_text_block(frame, [
            f"{args.episode_id}  frame={video_frame_idx}",
            "mesh: R=orange L=blue  boxes: mesh bbox  dots: mocap markers",
        ], line_height=25)
        writer.write(frame)

        for hand in ("left", "right"):
            crop = read_precrop(video_frame_idx, hand)
            if crop is None:  # Prevalidated above; protects against LMDB corruption.
                raise RuntimeError(
                    f"Precomputed crop vanished while reading frame {video_frame_idx}, "
                    f"hand={hand} from {crop_lmdb}")
            if crop.shape[:2] != (crop_size, crop_size):
                crop = cv2.resize(crop, (crop_size, crop_size))
            if hand in hand_verts:
                affine = _precrop_affine(
                    hand_data[hand]["keypoints"][global_i],
                    hand_data[hand]["keypoints_valid"][global_i],
                    T_W_C, K, dist, intrinsics_info, video_w, hand, crop_size)
                if affine is not None:
                    mesh_px, mesh_depth = vu.project_and_map(
                        hand_verts[hand], T_W_C, K, dist, intrinsics_info)
                    if hand == "left":
                        mesh_px[:, 0] = (video_w - 1) - mesh_px[:, 0]
                    mesh_crop_px = cv2.transform(
                        mesh_px.astype(np.float32).reshape(1, -1, 2), affine)[0]
                    vu.draw_wireframe(
                        crop, mesh_crop_px, mesh_depth, hand_faces[hand],
                        vu.HAND_COLORS_BGR[hand], args.line_width)
            cv2.putText(crop, f"{hand[0].upper()}  frame={video_frame_idx}",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, lineType=cv2.LINE_AA)
            crop_writers[hand].write(crop)

    writer.release()
    for crop_writer in crop_writers.values():
        crop_writer.release()
    if crop_env is not None:
        crop_env.close()
    if renderer is not None:
        renderer.delete()
    print(f"\nSaved: {output}")
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
    if args.add_video_fields or args.per_episode_crops_dir is not None:
        # video fields are required to look up per-episode crop frames.
        modalities.append("video_index")
    ds = vu.make_memmap_dataset(
        memmap_dir=args.memmap_dir, hand=args.hand,
        window_length=args.window_length, stride=args.stride,
        modalities=modalities,
        mano_npy_dir=args.mano_npy_dir,
        emg_field_preference="filtered_paper",
        emg_layout="emg2pose_interpolate16",
        emg2pose_channel_indices=[10, 12, 0, 1, 2, 4, 5, 6],
        channel_interpolate=False,
        norm_mode="per-dataset",
        norm_stats_path="./assets/per_dataset_norm_stats_repro_filtered_paper_alias.json",
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

        # Empirically the stored joint angles pair with the mirrored FK
        # profile for the right hand and the plain profile for the left
        # (see vu.fk_mesh_world).
        fk_verts, fk_faces = vu.skin_mesh_from_angles(
            joint_angles=ja_mid[:20], flip=(args.hand == "right"))
        fk_verts = fk_verts.copy()
        fk_faces = np.asarray(fk_faces)  # mirrored profile -> torch Tensor
        if args.hand == "right":
            fk_faces = fk_faces[:, [0, 2, 1]].copy()
        mano_verts, mano_faces = decoder.decode(
            mano_pose_mid, mano_beta, args.hand)
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


# ── parser ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--memmap-dir", type=Path,
                        default=Path("data/EgoEMG_full_memmap"))
    common.add_argument("--data-root", type=Path, default=Path("data"))
    common.add_argument("--allintra-root", type=Path,
                        default=Path("data/EgoEMG_videos"))
    common.add_argument("--allintra-suffix", default="_allintra")
    common.add_argument("--output-dir", type=Path, default=None)
    common.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    common.add_argument("--seed", type=int, default=42)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("vision", parents=[common],
                       description="Video replay: mesh + mocap markers + bbox "
                                   "projected on head-view frames -> MP4")
    p.add_argument("--episode-id", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--render-mode", default="wireframe",
                   choices=["wireframe", "mesh"])
    p.add_argument("--line-width", type=int, default=1)
    p.add_argument("--mesh-alpha", type=float, default=0.7)
    p.add_argument("--bbox-pad", type=int, default=15)
    p.add_argument("--mano-model-path", type=Path, default=None)
    p.add_argument("--calibration-json", type=Path, default=None)
    p.add_argument("--crops-dir", type=Path, default=Path("data/EgoEMG_crops"),
                   help="directory containing precomputed per-episode crop LMDBs")
    p.add_argument("--output", type=Path, default=None)

    p = sub.add_parser("timeline", parents=[common],
                       description="EMG / joint angles / MANO time series -> PNG")
    p.add_argument("--episode", type=int, default=3)
    p.add_argument("--hand", default="right", choices=["left", "right"])
    p.add_argument("--emg-preference", default="filtered_paper",
                   choices=["raw", "filtered", "filtered_paper"],
                   help="EMG field preference (left-hand filtered is absent "
                        "from the unified memmap; use filtered_paper)")
    p.add_argument("--offset", type=int, default=100000)
    p.add_argument("--window", type=int, default=2000)
    p.add_argument("--out-path", type=Path, default=None)

    p = sub.add_parser("mesh", parents=[common],
                       description="MANO/FK mesh overlay on head-view frames; "
                                   "GLB-only export with --glb-only")
    p.add_argument("--mano-model-path", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--line-width", type=int, default=1)
    p.add_argument("--render-mode", default="mesh",
                   choices=["wireframe", "mesh"])
    p.add_argument("--mesh-alpha", type=float, default=0.7)
    p.add_argument("--calibration-path", type=Path, default=None)
    p.add_argument("--glb-only", action="store_true",
                   help="skip videos: export MANO/FK world-space GLBs "
                        "(with mocap + MANO-surface markers) only")

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

    return parser


MODES = {
    "vision": run_vision_video,
    "timeline": run_timeline,
    "mesh": run_mesh,
    "fk_vs_mano": run_fk_vs_mano,
}


def main() -> int:
    vu.setup_headless_environment()
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = Path("/tmp/egoemg_viz") / args.mode
    return MODES[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
