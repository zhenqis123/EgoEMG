#!/usr/bin/env python3
"""
Random-sample visualization of batch IK results.

For each sampled frame, saves:
  - a GLB with MANO mesh (blue) + UmeTrack mesh (red) for 3D comparison
  - the head-view frame with crop rectangle overlay

Usage:
  python scripts/viz/visualize_ik_results.py --num-samples 8 --hand right
  python scripts/viz/visualize_ik_results.py --num-samples 16 --hand both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MANOTORCH_ROOT = Path("../manotorch")
if str(MANOTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MANOTORCH_ROOT))
from manotorch.manolayer import ManoLayer

MANO_ASSETS_ROOT = Path("../HandVQVAE/assets/mano")
MEMMAP_ROOT = Path("data/EgoEMG_memmap")
FLIP_MATRIX = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
ALIGN_SCALE = 1.0843137502670288
ALIGN_TRANS = np.array([106.72334, -11.8804455, -4.48328], dtype=np.float32)


def load_mm(manifest, mm_dir, name):
    info = manifest["fields"][name]
    return np.memmap(
        f"{mm_dir}/{info['filename']}",
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def load_ep_mm(manifest, mm_dir, name):
    info = manifest["episode_fields"][name]
    return np.memmap(
        f"{mm_dir}/{info['filename']}",
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def save_glb(path, mano_verts, mano_faces, ut_verts, ut_faces,
             mano_color=(70, 130, 180, 200), ut_color=(220, 80, 60, 200)):
    import trimesh
    ma = trimesh.Trimesh(vertices=mano_verts, faces=mano_faces[:, ::-1], process=False)
    ma.visual.vertex_colors = np.tile(mano_color, (len(mano_verts), 1)).astype(np.uint8)
    mb = trimesh.Trimesh(vertices=ut_verts, faces=ut_faces, process=False)
    mb.visual.vertex_colors = np.tile(ut_color, (len(ut_verts), 1)).astype(np.uint8)
    trimesh.Scene([ma, mb]).export(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-root", type=Path, default=MEMMAP_ROOT)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--hand", default="both", choices=["left", "right", "both"])
    parser.add_argument("--output", type=Path, default=Path("ik_vis"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data/EgoEMG",
                        help="Root of EgoEMG video data (for head-view frames).")
    parser.add_argument("--allintra-root",
                        default="data/EgoEMG_videos",
                        help="Root of all-intra re-encoded head-view videos.")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ── Load manifest & metadata ────────────────────────────────────────
    with open(args.memmap_root / "manifest.json") as f:
        manifest = json.load(f)
    md = np.load(args.memmap_root / "metadata.npz", allow_pickle=False)
    total_rows = int(manifest["total_rows"])

    # Check that IK output fields exist
    hands = ["right", "left"] if args.hand == "both" else [args.hand]
    for h in hands:
        field = f"generated_joint_angles_{h}"
        if field not in manifest["fields"]:
            raise KeyError(f"Field '{field}' not in manifest. Run batch_ik_mesh.py first.")

    # ── Init models ─────────────────────────────────────────────────────
    from egoemg.kinematics import apply_to_hand_model, load_default_hand_model

    mano_layer = ManoLayer(
        rot_mode="axisang", side="right",
        mano_assets_root=str(MANO_ASSETS_ROOT),
        use_pca=False, flat_hand_mean=False,
    ).to(device)

    hand_model = load_default_hand_model()
    hand_model = apply_to_hand_model(hand_model, lambda t: t.float().to(device))
    mano_faces = mano_layer.th_faces.cpu().numpy()
    ut_faces = hand_model.mesh_triangles.cpu().numpy()

    flip_t = torch.from_numpy(FLIP_MATRIX).float().to(device)

    trans_init = torch.tensor(ALIGN_TRANS.tolist(), dtype=torch.float32, device=device)
    fixed_scale = ALIGN_SCALE

    # ── Sample random frames ────────────────────────────────────────────
    n_samples = min(args.num_samples, total_rows)
    frame_indices = sorted(rng.choice(total_rows, size=n_samples, replace=False))
    print(f"Sampled {n_samples} frames: {frame_indices}")

    # ── Setup video readers ─────────────────────────────────────────────
    ep_idx_mm = load_mm(manifest, args.memmap_root, "episode_index")
    frame_idx_mm = load_mm(manifest, args.memmap_root, "image_head_frame_index")
    video_paths = _decode_bytes(md["episode_head_video_path"])
    ep_start_idx = np.asarray(md["episode_start_idx"], dtype=np.int64)
    ep_end_idx = np.asarray(md["episode_end_idx"], dtype=np.int64)
    beta_idx_arr = md["episode_beta_idx"]

    # Pre-load pose, angles, beta memmaps
    pose_mms = {}
    angle_mms = {}
    beta_mms = {}
    for h in hands:
        pose_mms[h] = load_mm(manifest, args.memmap_root, f"generated_mano_{h}_pose")
        angle_mms[h] = load_mm(manifest, args.memmap_root, f"generated_joint_angles_{h}")
        beta_mms[h] = load_ep_mm(manifest, args.memmap_root, f"generated_mano_{h}_beta")

    # UmeTrack FK (single-frame, for visualization)
    def umetrack_fk_single(angles_20):
        from egoemg.kinematics import broadcast_hand_model_to
        from egoemg.UmeTrack.lib.common.hand_skinning import (
            _get_skinned_vertices, _hand_skinning_transform, _lbs,
        )
        hm = broadcast_hand_model_to(hand_model, (1,))
        hm = apply_to_hand_model(hm, lambda t: t.float())
        wrist_tf = torch.eye(4, device=device).unsqueeze(0)
        a = torch.cat([angles_20, torch.zeros(2, device=device)]).reshape(1, -1)
        skin_xfs = _hand_skinning_transform(
            hm.joint_rotation_axes.reshape(1, -1, 3),
            hm.joint_rest_positions.reshape(1, -1, 3),
            a, wrist_tf,
        )
        w = hm.dense_bone_weights.reshape(1, -1, 17)
        mr = hm.mesh_vertices.reshape(1, -1, 3)
        v = _get_skinned_vertices(mr, w)
        mesh = _lbs(skin_xfs, v)[..., :3][0]
        return mesh

    # ── Process each frame ──────────────────────────────────────────────
    for fi in frame_indices:
        ep = int(ep_idx_mm[fi])
        print(f"\nFrame {fi} (episode {ep})")

        # Determine episode-specific frame index
        ep_s = int(ep_start_idx[ep])
        ep_e = int(ep_end_idx[ep])
        local_fi = fi - ep_s

        for h in hands:
            print(f"  Hand: {h}")

            # ── MANO FK ─────────────────────────────────────────────────
            pose_np = pose_mms[h][fi].astype(np.float32).copy()
            beta_idx = int(beta_idx_arr[ep])
            beta_np = beta_mms[h][beta_idx].astype(np.float32).copy()
            pose_t = torch.from_numpy(pose_np).unsqueeze(0).to(device)
            beta_t = torch.from_numpy(beta_np).unsqueeze(0).to(device)

            with torch.no_grad():
                out = mano_layer(pose_t, beta_t)
                mano_v = out.verts[0] * 1000.0  # (778, 3) in mm

            # ── UmeTrack FK ──────────────────────────────────────────────
            angles_np = angle_mms[h][fi].astype(np.float32).copy()
            angles_t = torch.from_numpy(angles_np).to(device)

            with torch.no_grad():
                ut_mesh = umetrack_fk_single(angles_t)

            # Transform MANO into UmeTrack frame (same as IK optimization)
            mano_v_aligned = fixed_scale * (mano_v @ flip_t.T) + trans_init

            # ── Save GLB ─────────────────────────────────────────────────
            glb_path = args.output / f"frame_{fi:08d}_{h}.glb"
            save_glb(glb_path,
                     mano_v_aligned.cpu().numpy(), mano_faces,
                     ut_mesh.cpu().numpy(), ut_faces)
            print(f"    GLB: {glb_path}")

        # ── Extract webcam frame ─────────────────────────────────────────
        vp = str(Path(args.data_root) / _decode_single(video_paths[ep]))
        allintra_vp = _resolve_allintra(vp, args.data_root, args.allintra_root)

        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(allintra_vp), ctx=cpu(0))
            device_frame_idx = int(frame_idx_mm[fi])
            vfi = max(0, min(device_frame_idx, len(vr) - 1))
            frame_rgb = vr[vfi].asnumpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Overlay frame info
            cv2.putText(frame_bgr, f"frame={fi} ep={ep} local={local_fi} vfi={vfi}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            img_path = args.output / f"frame_{fi:08d}_webcam.jpg"
            cv2.imwrite(str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Webcam: {img_path}")
        except Exception as e:
            print(f"  Webcam: SKIP ({e})")

    print(f"\nDone. Output in {args.output}")


def _decode_bytes(values):
    return [v.decode("utf-8", errors="replace").rstrip("\x00")
            if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values]


def _decode_single(v):
    return v.decode("utf-8", errors="replace").rstrip("\x00") if isinstance(v, (bytes, np.bytes_)) else str(v)


def _resolve_allintra(raw_path, data_root, allintra_root):
    raw = Path(raw_path)
    if raw.is_absolute():
        try:
            rel = raw.relative_to(Path(data_root).resolve())
        except ValueError:
            rel = Path(raw.name)
    else:
        rel = raw
    return allintra_root / rel.with_name(f"{rel.stem}_allintra.mp4")


if __name__ == "__main__":
    main()
