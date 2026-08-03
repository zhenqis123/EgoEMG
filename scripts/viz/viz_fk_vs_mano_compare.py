#!/usr/bin/env python3
"""Compare FK mesh (UmeTrack) vs MANO mesh for EgoEMG dataset samples.

For each sampled frame, exports:
  - A GLB scene with FK mesh (blue, left) + MANO mesh (green, right)
  - The cropped hand image (if --per-episode-crops-dir is set)

Usage:
    python scripts/viz/viz_fk_vs_mano_compare.py \
        --memmap-dir data/EgoEMG_memmap \
        --mano-npy-dir data/EgoEMG/mano/chunk-000 \
        --per-episode-crops-dir /path/to/EgoEMG_crops \
        --out-dir /tmp/fk_vs_mano \
        --num-samples 10 \
        --hand right

    # Without crop images:
    python scripts/viz/viz_fk_vs_mano_compare.py \
        --memmap-dir data/EgoEMG_memmap \
        --mano-npy-dir data/EgoEMG/mano/chunk-000 \
        --out-dir /tmp/fk_vs_mano \
        --num-samples 10
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from egoemg.UmeTrack.lib.common.hand import HandModel
from egoemg.UmeTrack.lib.common.hand_skinning import _skin_points
from egoemg.UmeTrack.lib.tracker.video_pose_data import load_hand_model_from_dict
import json

MANO_ASSETS_ROOT = Path("../HandVQVAE/assets/mano")


# ── FK mesh skinning (same logic as egoemg.visualization.skin_mesh_from_angles) ─

def _load_default_hand_model() -> HandModel:
    umetrack_dir = Path(__file__).resolve().parent.parent / "emg2pose" / "UmeTrack"
    path = umetrack_dir / "dataset" / "generic_hand_model.json"
    with open(path) as f:
        hand_model_dict = json.load(f)
    return load_hand_model_from_dict(hand_model_dict)


def _mirror_profile(profile: HandModel) -> HandModel:
    mirrored_joint_rotation_axes = profile.joint_rotation_axes.clone()
    mirrored_joint_rest_positions = profile.joint_rest_positions.clone()
    mirrored_mesh_vertices = profile.mesh_vertices.clone() if profile.mesh_vertices is not None else None
    mirrored_joint_rotation_axes[..., 1:] *= -1
    mirrored_joint_rest_positions[..., 0] *= -1
    if mirrored_mesh_vertices is not None:
        mirrored_mesh_vertices[..., 0] *= -1
    return profile._replace(
        joint_rotation_axes=mirrored_joint_rotation_axes,
        joint_rest_positions=mirrored_joint_rest_positions,
        mesh_vertices=mirrored_mesh_vertices,
    )


def skin_mesh_from_angles(joint_angles, user_profile=None, flip=False):
    if user_profile is None:
        user_profile = _load_default_hand_model()
    if flip:
        user_profile = _mirror_profile(user_profile)

    joint_angles_t = torch.from_numpy(np.asarray(joint_angles)).float()
    leading_dims = joint_angles_t.shape[:-1]
    wrist_transforms = torch.broadcast_to(
        torch.eye(4), leading_dims + (4, 4),
    )
    vertices = _skin_points(
        user_profile.joint_rest_positions,
        user_profile.joint_rotation_axes,
        user_profile.dense_bone_weights,
        joint_angles_t,
        user_profile.mesh_vertices,
        wrist_transforms,
    )
    vertices = vertices.reshape(list(leading_dims) + list(vertices.shape[-2:]))
    triangles = user_profile.mesh_triangles
    return vertices.cpu().numpy(), triangles.cpu().numpy()


# ── MANO mesh ─────────────────────────────────────────────────────────────────

class ManoMeshDecoder:
    """Reusable MANO mesh decoder. Create once, call per frame."""

    def __init__(self, device: torch.device):
        from manotorch.manolayer import ManoLayer

        self.mano_layer = ManoLayer(
            use_pca=False,
            mano_assets_root=str(MANO_ASSETS_ROOT),
            flat_hand_mean=False,
        ).to(device)
        self.device = device
        self._faces = None

    def decode(self, pose: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose_t = torch.from_numpy(np.asarray(pose).copy()).float().to(self.device).unsqueeze(0)
        beta_t = torch.from_numpy(np.asarray(beta).copy()).float().to(self.device).unsqueeze(0)
        pose_t[:, :3] = 0.0  # zero global orient

        with torch.no_grad():
            out = self.mano_layer(pose_t, beta_t)
        verts = out.verts[0].cpu().numpy()
        if self._faces is None:
            self._faces = self.mano_layer.th_faces.cpu().numpy()
        return verts, self._faces


# ── Crop image loading ────────────────────────────────────────────────────────

def load_crop_image(
    crops_dir: Path,
    episode_id: str,
    frame_idx: int,
    hand_code: str,
) -> np.ndarray | None:
    """Load a single crop image from per-episode LMDB."""
    import lmdb
    from PIL import Image

    lmdb_path = crops_dir / f"{episode_id}.lmdb"
    if not lmdb_path.exists():
        return None
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        key = f"{frame_idx:08d}_{hand_code}".encode()
        jpeg_bytes = txn.get(key)
        if jpeg_bytes is None:
            return None
        img = Image.open(io.BytesIO(jpeg_bytes))
        return np.asarray(img)


# ── GLB export ────────────────────────────────────────────────────────────────

def save_comparison_glb(
    out_path: Path,
    fk_verts: np.ndarray,
    fk_faces: np.ndarray,
    mano_verts: np.ndarray,
    mano_faces: np.ndarray,
    fk_color: tuple[int, int, int, int] = (80, 140, 240, 220),
    mano_color: tuple[int, int, int, int] = (80, 220, 140, 220),
    offset_x: float = 0.15,
):
    """Export GLB with FK mesh (left, blue) and MANO mesh (right, green)."""
    import trimesh

    parts = []

    # FK mesh (left)
    fk_v = fk_verts.copy()
    fk_v[:, 0] -= offset_x
    fk_mesh = trimesh.Trimesh(vertices=fk_v, faces=fk_faces, process=False)
    fk_mesh.visual.vertex_colors = fk_color
    parts.append(fk_mesh)

    # MANO mesh (right)
    mano_v = mano_verts.copy()
    mano_v[:, 0] += offset_x
    mano_mesh = trimesh.Trimesh(vertices=mano_v, faces=mano_faces, process=False)
    mano_mesh.visual.vertex_colors = mano_color
    parts.append(mano_mesh)

    # Label spheres
    fk_label_y = float(fk_verts[:, 1].max()) + 0.03
    label = trimesh.creation.icosphere(subdivisions=2, radius=0.008)
    label.apply_translation([-offset_x, fk_label_y, 0.0])
    label.visual.vertex_colors = fk_color
    parts.append(label)

    mano_label_y = float(mano_verts[:, 1].max()) + 0.03
    label_m = trimesh.creation.icosphere(subdivisions=2, radius=0.008)
    label_m.apply_translation([offset_x, mano_label_y, 0.0])
    label_m.visual.vertex_colors = mano_color
    parts.append(label_m)

    scene = trimesh.Scene(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memmap-dir", type=Path, required=True)
    parser.add_argument("--mano-npy-dir", type=Path, required=True)
    parser.add_argument("--per-episode-crops-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/fk_vs_mano"))
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--window-length", type=int, default=7790)
    parser.add_argument("--stride", type=int, default=7790)
    parser.add_argument("--add-video-fields", action="store_true",
                        help="Load image_webcam_frame_index for richer filenames.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.out_dir
    gbl_dir = out_dir / "glb"
    crop_dir = out_dir / "crops"
    gbl_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

    modalities = ["emg", "joint_angles", "mano", "labels"]
    if args.add_video_fields:
        modalities.append("video_index")

    print(f"Loading EgoEMG dataset from {args.memmap_dir} ...")
    ds = EgoEmgMemmapDataset(
        memmap_dir=args.memmap_dir,
        window_length=args.window_length,
        stride=args.stride,
        modalities=modalities,
        target_hand=args.hand,
        emg_field_preference="filtered",
        emg_layout="emg2pose_interpolate16",
        emg2pose_channel_indices=[10, 12, 0, 1, 2, 4, 5, 6],
        channel_interpolate=False,
        norm_mode="per-dataset",
        norm_stats_path="./assets/per_dataset_norm_stats.json",
        dataset_name="egoemg",
        mano_npy_dir=args.mano_npy_dir,
        jitter=False,
    )
    print(f"  {len(ds)} windows, {len(ds._episode_id)} episodes")

    # ── Init models (once) ────────────────────────────────────────────────
    print("Initializing MANO decoder ...")
    mano_decoder = ManoMeshDecoder(device)
    print("  MANO decoder ready")

    # ── Sample frames ─────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    if args.episode is not None:
        ep_mask = ds._block_episode_idx == args.episode
        valid_indices = np.where(ep_mask)[0]
    else:
        valid_indices = np.arange(len(ds))

    if len(valid_indices) == 0:
        print("No windows found!")
        return

    n = min(args.num_samples, len(valid_indices))
    chosen = rng.choice(valid_indices, size=n, replace=False)
    hand_code = "L" if args.hand == "left" else "R"

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

        # FK mesh (same angle semantics as egoemg.visualization)
        fk_verts, fk_faces = skin_mesh_from_angles(
            joint_angles=ja_mid[:20],
            flip=(args.hand == "left"),
        )
        fk_verts = fk_verts.copy()

        # MANO mesh
        mano_verts, mano_faces = mano_decoder.decode(mano_pose_mid, mano_beta)

        # Scale FK mesh to match MANO
        fk_span = fk_verts.max(axis=0) - fk_verts.min(axis=0)
        mano_span = mano_verts.max(axis=0) - mano_verts.min(axis=0)
        if np.median(fk_span) > 1e-6:
            scale_factor = float(np.median(mano_span) / np.median(fk_span))
            fk_verts = fk_verts * scale_factor

        # Export GLB (use window_start_idx as unique frame identifier)
        win_start = int(np.asarray(sample["window_start_idx"]))
        gbl_path = gbl_dir / f"{ep_id}_{args.hand}_win{win_start:08d}.glb"
        save_comparison_glb(gbl_path, fk_verts, fk_faces, mano_verts, mano_faces)
        print(f"  [{i+1}/{n}] {gbl_path.name}")

        # Save crop image
        if args.per_episode_crops_dir is not None:
            # For crop LMDB, we need the actual video frame index, not window start.
            # Pre-crops use the image_webcam_frame_index as key.
            if "image_webcam_frame_index" in sample:
                vid_frame = int(np.asarray(sample["image_webcam_frame_index"])[center])
            else:
                vid_frame = win_start  # fallback
            crop_img = load_crop_image(args.per_episode_crops_dir, ep_id, vid_frame, hand_code)
            if crop_img is not None:
                from PIL import Image
                crop_path = crop_dir / f"{ep_id}_{args.hand}_win{win_start:08d}.png"
                Image.fromarray(crop_img).save(str(crop_path))
                print(f"         crop: {crop_path.name}")
            else:
                print(f"         crop: not found for frame {vid_frame}")

    print(f"\nDone. GLB files: {gbl_dir}")
    if args.per_episode_crops_dir is not None:
        print(f"Crop images: {crop_dir}")


if __name__ == "__main__":
    main()
