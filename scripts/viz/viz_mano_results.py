"""
Visualize MANO inference results: Kabsch-aligned mesh + GT keypoints.

Usage:
    python scripts/viz/viz_mano_results.py --episode 0 --device cuda:5 --ckpt v45 --num-frames 5
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

WILOR_ROOT = Path("/home/xiziheng/develop/WiLoR")
EGOEMG_ROOT = Path("/home/xiziheng/develop/emg2pose")
if str(WILOR_ROOT) not in sys.path:
    sys.path.insert(0, str(WILOR_ROOT))

from manotorch.manolayer import ManoLayer
from markers2mano.geometry import transform_joints_coordinates_torch
from markers2mano.graph_transformer import EfficientGraphTransformer, six_d_to_rot_matrix
from markers2mano.geometry import matrix_to_axis_angle
from markers2mano.rigid_align import compute_aligned_error

MARKER_VERT_INDICES = torch.tensor(
    [191, 88, 253, 708, 729, 144, 87, 295, 319, 220,
     365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673],
    dtype=torch.long,
)

CKPT_PATHS = {
    "v43": WILOR_ROOT / "tb_logs/m2m_pose_shape_run/version_43/checkpoints/last-epoch=499.ckpt",
    "v44": WILOR_ROOT / "tb_logs/m2m_pose_shape_run/version_44/checkpoints/last-epoch=99.ckpt",
    "v45": WILOR_ROOT / "tb_logs/m2m_pose_shape_run/version_45/checkpoints/last-epoch=99.ckpt",
}

DATA_DIR = EGOEMG_ROOT / "data" / "EgoEMG" / "data" / "chunk-000"
VIZ_DIR = EGOEMG_ROOT / "data" / "EgoEMG" / "mano_viz" / "chunk-000"


def col_to_np(col):
    col = col.combine_chunks(); ct = col.type
    if pa.types.is_list(ct):
        inner = ct.value_type
        if pa.types.is_list(inner):
            flat = col.flatten().flatten().to_numpy(zero_copy_only=False)
            first = col[0].as_py()
            if first and len(first) > 0 and len(first[0]) > 0:
                return flat.reshape(-1, len(first), len(first[0]))
        return np.asarray(col.to_pylist())
    return col.to_numpy(zero_copy_only=False)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EfficientGraphTransformer(
        num_markers=21, beta_dim=10, embed_dim=1280,
        num_layers=12, num_heads=8, dropout=0.0, ffn_ratio=4,
    )
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def infer_frame(model, mano, kp_21, device, flip_z=False):
    """Run model on a single frame's keypoints."""
    kpt = torch.from_numpy(kp_21[None]).float().to(device)
    rooted = kpt - kpt[:, 0:1, :]
    local, T = transform_joints_coordinates_torch(rooted)
    if flip_z:
        local[:, :, 2] = -local[:, :, 2]

    with torch.no_grad():
        pose_6d, beta = model(local)
    pose_mat = six_d_to_rot_matrix(pose_6d.view(-1, 16, 6))
    pose_ax = matrix_to_axis_angle(pose_mat).view(-1, 48)
    pose_ax[:, :3] = 0.0

    with torch.no_grad():
        mano_out = mano(pose_ax, beta)
        markers_local = mano_out.verts[0, MARKER_VERT_INDICES.to(device)]

    # Kabsch alignment: find R,t to map MANO markers -> GT world keypoints
    gt_world = kpt[0]
    err, R, t = compute_aligned_error(markers_local, gt_world)
    # Align: markers_world = markers_local @ R.T + t
    markers_world = markers_local @ R.T + t

    return {
        "pose_ax": pose_ax[0].cpu().numpy(),
        "beta": beta[0].cpu().numpy(),
        "markers_local": markers_local.cpu().numpy(),
        "markers_world": markers_world.cpu().numpy(),
        "kabsch_R": R.cpu().numpy(),
        "kabsch_t": t.cpu().numpy(),
        "aligned_err_mm": err.mean().item() * 1000,
        "T_world2local": T.cpu().numpy(),
    }


def save_glb(out_path, verts_world, faces, markers_world, gt_keypoints):
    """Save a GLB scene: mesh + predicted markers (yellow) + GT keypoints (red)."""
    import trimesh

    # Mesh in world space
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    mesh.visual.vertex_colors = [80, 200, 120, 220]  # green

    # Predicted markers
    pred_spheres = []
    for pt in markers_world:
        s = trimesh.creation.icosphere(subdivisions=2, radius=0.004)
        s.apply_translation(pt)
        s.visual.vertex_colors = [255, 215, 0, 255]  # gold
        pred_spheres.append(s)

    # GT keypoints
    gt_spheres = []
    for pt in gt_keypoints:
        s = trimesh.creation.icosphere(subdivisions=2, radius=0.003)
        s.apply_translation(pt)
        s.visual.vertex_colors = [255, 50, 50, 200]  # red
        gt_spheres.append(s)

    scene = trimesh.Scene([mesh] + pred_spheres + gt_spheres)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--ckpt", choices=list(CKPT_PATHS.keys()), default="v45")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Number of frames to visualize per hand")
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    args = parser.parse_args()

    device = torch.device(args.device)
    viz_dir = args.viz_dir / f"viz_{args.ckpt}_ep{args.episode}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    parquet_files = sorted(
        f for f in DATA_DIR.iterdir()
        if f.suffix == ".parquet" and "_bak" not in f.name
    )
    if args.episode >= len(parquet_files):
        raise IndexError(f"Episode {args.episode} out of range (found {len(parquet_files)})")
    ep_file = parquet_files[args.episode]
    stem = ep_file.stem
    print(f"Loading {ep_file.name}...")

    table = pq.read_table(ep_file)
    kp_l = col_to_np(table.column("observation.mocap.hand.left.keypoints")).astype(np.float32)
    kp_r = col_to_np(table.column("observation.mocap.hand.right.keypoints")).astype(np.float32)
    valid_l = col_to_np(table.column("observation.mocap.hand.left.valid")).astype(bool)
    valid_r = col_to_np(table.column("observation.mocap.hand.right.valid")).astype(bool)

    # Load model
    model = load_model(CKPT_PATHS[args.ckpt], device)
    mano = ManoLayer(use_pca=False, mano_assets_root="/home/xiziheng/develop/HandVQVAE/assets/mano",
                     flat_hand_mean=False).to(device)
    faces = mano.get_mano_closed_faces().detach().cpu().numpy()

    # Select frames to visualize (evenly spaced among valid frames)
    for hand, kp, valid, flip_z in [
        ("right", kp_r, valid_r, False),
        ("left", kp_l, valid_l, True),
    ]:
        valid_indices = np.where(valid.any(axis=1))[0]
        if len(valid_indices) == 0:
            print(f"  No valid frames for {hand}")
            continue

        chosen = valid_indices[np.linspace(0, len(valid_indices) - 1, args.num_frames, dtype=int)]

        for fi in chosen:
            result = infer_frame(model, mano, kp[fi], device, flip_z=flip_z)

            # Get full MANO mesh in world space using Kabsch R,t
            with torch.no_grad():
                pose_t = torch.from_numpy(result["pose_ax"][None]).float().to(device)
                beta_t = torch.from_numpy(result["beta"][None]).float().to(device)
                mano_out = mano(pose_t, beta_t)
                verts_local = mano_out.verts[0].detach().cpu().numpy()

            # Transform to world: V_world = V_local @ R.T + t
            verts_world = verts_local @ result["kabsch_R"].T + result["kabsch_t"]

            out_path = viz_dir / f"{stem}_{hand}_frame_{fi:06d}_err{result['aligned_err_mm']:.1f}mm.glb"
            save_glb(out_path, verts_world, faces, result["markers_world"], kp[fi])

            print(f"  {hand} frame {fi:6d}: aligned_err={result['aligned_err_mm']:.1f}mm  ->  {out_path.name}")

    print(f"\nDone! Saved to {viz_dir}")


if __name__ == "__main__":
    main()
