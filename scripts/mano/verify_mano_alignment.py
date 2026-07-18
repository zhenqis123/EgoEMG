"""
Verify MANO alignment: load inferred pose/beta/trans + GT keypoints,
compute before/after alignment marker errors, and optionally visualize.

Usage:
    python scripts/mano/verify_mano_alignment.py \
        --episode 0 \
        --hand right \
        --mano-dir data/EgoEMG/mano/chunk-000 \
        --data-dir data/EgoEMG/data/chunk-000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq

WILOR_ROOT = Path("/home/xiziheng/develop/WiLoR")
if str(WILOR_ROOT) not in sys.path:
    sys.path.append(str(WILOR_ROOT))

from manotorch.manolayer import ManoLayer
from markers2mano.rigid_align import compute_aligned_error

MARKER_VERT_INDICES = torch.tensor(
    [191, 88, 253, 708, 729, 144, 87, 295, 319, 220,
     365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673],
    dtype=torch.long,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--mano-dir", type=Path,
                        default=Path("data/EgoEMG/mano/chunk-000"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/EgoEMG/data/chunk-000"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=20,
                        help="Max sampled frames to verify")
    parser.add_argument("--viz", action="store_true",
                        help="Save GLB visualization")
    args = parser.parse_args()

    device = torch.device(args.device)
    hand = args.hand

    # --- Load inferred MANO params ---
    mano_dir = args.mano_dir
    episode_files = sorted(f for f in mano_dir.iterdir()
                           if f.name.endswith(".parquet") or f.name.endswith(".npy"))
    # Find the episode file by index
    parquet_files = sorted(f for f in args.data_dir.iterdir() if f.suffix == ".parquet")
    if args.episode >= len(parquet_files):
        raise IndexError(f"Episode {args.episode} out of range (found {len(parquet_files)})")

    ep_file = parquet_files[args.episode]
    stem = ep_file.stem

    pose_path = mano_dir / f"{stem}_{hand}_pose.npy"
    beta_path = mano_dir / f"{stem}_{hand}_beta.npy"
    trans_path = mano_dir / f"{stem}_{hand}_trans.npy"

    if not pose_path.exists():
        print(f"Error: {pose_path} not found. Run infer_mano_for_egoemg.py first.")
        sys.exit(1)

    pose_full = np.load(pose_path)
    beta_mean = np.load(beta_path)
    trans = np.load(trans_path) if trans_path.exists() else np.zeros(3, dtype=np.float32)

    print(f"Episode: {stem}, hand={hand}")
    print(f"  pose shape: {pose_full.shape}")
    print(f"  beta: {beta_mean}")
    print(f"  trans: {trans}")

    # --- Load GT keypoints ---
    table = pq.read_table(ep_file)
    col_key = f"observation.mocap.hand.{hand}.keypoints"
    col_valid = f"observation.mocap.hand.{hand}.valid"

    # Simple parquet column to numpy (same logic as infer script)
    def col_to_numpy(col):
        col = col.combine_chunks()
        col_type = col.type
        import pyarrow as pa
        if pa.types.is_list(col_type):
            inner_type = col_type.value_type
            if pa.types.is_list(inner_type):
                flat = col.flatten().flatten().to_numpy(zero_copy_only=False)
                first_val = col[0].as_py()
                if first_val and len(first_val) > 0 and len(first_val[0]) > 0:
                    return flat.reshape(-1, len(first_val), len(first_val[0]))
            return np.asarray(col.to_pylist())
        return col.to_numpy(zero_copy_only=False)

    kp = col_to_numpy(table.column(col_key)).astype(np.float32)
    valid = col_to_numpy(table.column(col_valid)).astype(bool)
    frame_valid = valid.any(axis=1) if valid.ndim == 2 else np.ones(len(kp), dtype=bool)
    T = len(kp)
    print(f"  GT keypoints shape: {kp.shape}")

    # --- Select frames to verify ---
    valid_indices = np.where(frame_valid)[0]
    stride = max(1, T // len(valid_indices)) if len(valid_indices) > 0 else 100
    # Reconstruct the sampling used in inference
    sampled_indices = valid_indices[::stride] if len(valid_indices) > 0 else np.arange(0, T, stride)
    if len(sampled_indices) == 0:
        sampled_indices = np.array([0])
    verify_indices = sampled_indices[:args.max_frames]

    # --- Load MANO ---
    mano = ManoLayer(
        use_pca=False,
        mano_assets_root="/home/xiziheng/develop/HandVQVAE/assets/mano",
        flat_hand_mean=False,
    ).to(device)

    marker_idx = MARKER_VERT_INDICES.to(device)

    # --- Compute errors ---
    before_errors = []
    after_errors = []

    for idx in verify_indices:
        pose = torch.from_numpy(pose_full[idx:idx + 1]).float().to(device)
        beta = torch.from_numpy(beta_mean[None]).float().to(device)
        gt_marker = torch.from_numpy(kp[idx]).float().to(device)  # (21, 3)

        with torch.no_grad():
            mano_out = mano(pose, beta)
            pred_markers = mano_out.verts[:, marker_idx, :]  # (1, 21, 3) local

        # Before alignment: L2 from local to world (no transform)
        before_err = torch.norm(pred_markers[0] - gt_marker, dim=-1).mean().item() * 1000
        before_errors.append(before_err)

        # After alignment: + trans
        aligned = pred_markers[0] + torch.from_numpy(trans).to(device)
        after_err = torch.norm(aligned - gt_marker, dim=-1).mean().item() * 1000
        after_errors.append(after_err)

    print(f"\n=== Verification Results ({len(verify_indices)} frames) ===")
    print(f"  Before alignment (local → world, no transform): {np.mean(before_errors):.2f} ± {np.std(before_errors):.2f} mm")
    print(f"  After alignment  (+ trans):                     {np.mean(after_errors):.2f} ± {np.std(after_errors):.2f} mm")

    # Also compute optimal Kabsch per frame for reference
    kabsch_errors = []
    for idx in verify_indices:
        pose = torch.from_numpy(pose_full[idx:idx + 1]).float().to(device)
        beta = torch.from_numpy(beta_mean[None]).float().to(device)
        gt_marker = torch.from_numpy(kp[idx]).float().to(device)

        with torch.no_grad():
            mano_out = mano(pose, beta)
            pred_markers = mano_out.verts[:, marker_idx, :]

        err_per_marker, _, _ = compute_aligned_error(pred_markers[0], gt_marker)
        kabsch_errors.append(err_per_marker.mean().item() * 1000)

    print(f"  Kabsch optimal (per-frame R+t):                 {np.mean(kabsch_errors):.2f} ± {np.std(kabsch_errors):.2f} mm")
    print(f"\n  Trans-only gap vs Kabsch: {np.mean(after_errors) - np.mean(kabsch_errors):.2f} mm")
    print(f"  (Small gap means pose/beta quality is good; large gap means residual rotation error)")

    # --- Visualization ---
    if args.viz:
        import trimesh
        faces = mano.get_mano_closed_faces().detach().cpu().numpy()
        viz_dir = Path(f"data/EgoEMG/mano_viz_verify/chunk-000")
        viz_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(verify_indices[:5]):
            pose = torch.from_numpy(pose_full[idx:idx + 1]).float().to(device)
            beta = torch.from_numpy(beta_mean[None]).float().to(device)
            gt_marker = torch.from_numpy(kp[idx]).float().to(device)

            with torch.no_grad():
                mano_out = mano(pose, beta)
                verts = mano_out.verts[0].detach().cpu().numpy()

            # Aligned verts
            verts_aligned = verts + trans

            mesh = trimesh.Trimesh(vertices=verts_aligned, faces=faces, process=False)
            mesh.visual.vertex_colors = [80, 200, 120, 220]

            spheres = []
            for pt in gt_marker.cpu().numpy():
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.004)
                sphere.apply_translation(pt)
                sphere.visual.vertex_colors = [255, 215, 0, 255]
                spheres.append(sphere)

            scene = trimesh.Scene([mesh] + spheres)
            out = viz_dir / f"{stem}_{hand}_frame_{idx:06d}_verify.glb"
            scene.export(out)
            print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
