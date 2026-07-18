"""
Infer MANO pose & beta for EgoEMG keypoints (single episode with frame range support).

This version supports specifying a single episode (via --episode-idx or --episode-name)
and optionally a frame range [start, end) with custom stride.

Left-hand inference uses the validated `flip_local_z` strategy after
`transform_joints_coordinates_torch`, while right-hand inference keeps the
original local coordinates.

Important semantic note:
- EgoEMG stores both left and right hand pose labels in a MANO-right canonical
  parameterization.
- `*_right_pose.npy` is decoded with `MANO_RIGHT` directly.
- `*_left_pose.npy` is also decoded with `MANO_RIGHT`; left-hand geometry is
  recovered for world-space mesh visualization by mirroring raw MANO geometry
  along x before applying the precomputed local-to-world transform.
- Do not decode EgoEMG `*_left_pose.npy` with `MANO_LEFT`.

Output layout:
    data/EgoEMG/mano/chunk-000/
        episode_XXXXXX_left_pose.npy   (T, 48) float32
        episode_XXXXXX_left_beta.npy   (10,) float32
        episode_XXXXXX_right_pose.npy  (T, 48) float32
        episode_XXXXXX_right_beta.npy  (10,) float32

Optional visualization layout:
    data/EgoEMG/mano_viz/chunk-000/
        episode_XXXXXX_left_frame_YYYYYY.glb
        episode_XXXXXX_right_frame_YYYYYY.glb

Usage:
    # Process a single episode by index, all frames, stride=100
    python infer_mano_single.py --episode-idx 0 --device 0

    # Process a single episode by name, frames 500-2000, stride 10
    python infer_mano_single.py --episode-name "episode_000123.parquet" \
        --start-frame 500 --end-frame 2000 --frame-stride 10 --device 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

WILOR_ROOT = Path("/home/xiziheng/develop/WiLoR")
if str(WILOR_ROOT) not in sys.path:
    sys.path.append(str(WILOR_ROOT))

from manotorch.manolayer import ManoLayer
from markers2mano.geometry import (
    matrix_to_axis_angle,
    transform_hand_coordinates_torch,
    transform_joints_coordinates_torch,
)
from markers2mano.graph_transformer import EfficientGraphTransformer, six_d_to_rot_matrix

EGOEMG_ROOT = Path("/home/xiziheng/develop/emg2pose/data/EgoEMG")
CKPT_PATH = Path(
    "/home/xiziheng/develop/WiLoR/tb_logs/m2m_pose_shape_run/version_45/checkpoints/last-epoch=99.ckpt"
)
MANO_ASSETS_ROOT = Path("/home/xiziheng/develop/HandVQVAE/assets/mano")
DATA_DIR = EGOEMG_ROOT / "data" / "chunk-000"
MANO_DIR = EGOEMG_ROOT / "mano" / "chunk-000"
VIZ_DIR = EGOEMG_ROOT / "mano_viz" / "chunk-000"

MARKER_VERT_INDICES = torch.tensor(
    [191, 88, 253, 708, 729, 144, 87, 295, 319, 220, 365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673],
    dtype=torch.long,
)
MIRROR_X_3 = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32)
MIRROR_X_33 = torch.diag(MIRROR_X_3)
MIRROR_Z_3 = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float32)

READ_COLS = [
    "observation.mocap.hand.left.keypoints",
    "observation.mocap.hand.right.keypoints",
    "observation.mocap.hand.left.valid",
    "observation.mocap.hand.right.valid",
]


def load_model(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    hidden_dim = hp.get("hidden_dim", 1280)
    num_layers = hp.get("num_layers", 12)
    heads = hp.get("heads", 8)

    model = EfficientGraphTransformer(
        num_markers=21,
        beta_dim=10,
        embed_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=heads,
        dropout=0.0,
        ffn_ratio=4,
    )
    state_dict = ckpt["state_dict"]
    cleaned = {
        k[len("model.") :] if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device).eval()
    print(f"Model loaded: dim={hidden_dim}, layers={num_layers}, heads={heads}")
    return model


def load_mano_layer(device: torch.device) -> ManoLayer:
    mano = ManoLayer(
        use_pca=False,
        mano_assets_root=str(MANO_ASSETS_ROOT),
        flat_hand_mean=False,
    )
    return mano.to(device)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda:"):
        return torch.device(device_arg)
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}")
    return torch.device(device_arg)


def _col_to_numpy(col: pa.ChunkedArray) -> np.ndarray:
    """Fast path: combine_chunks + flatten (avoids to_pylist, ~900x faster)."""
    col = col.combine_chunks()
    col_type = col.type
    if pa.types.is_list(col_type):
        inner_type = col_type.value_type
        if pa.types.is_list(inner_type):
            flat = col.flatten().flatten().to_numpy(zero_copy_only=False)
            first_val = col[0].as_py()
            if first_val and len(first_val) > 0 and len(first_val[0]) > 0:
                return flat.reshape(-1, len(first_val), len(first_val[0]))
            return np.asarray(col.to_pylist())
        flat = col.flatten().to_numpy(zero_copy_only=False)
        first_val = col[0].as_py()
        if first_val and len(first_val) > 0:
            return flat.reshape(-1, len(first_val))
        return flat
    return col.to_numpy(zero_copy_only=False)


def read_episode(fpath: Path) -> dict[str, Any]:
    t = pq.read_table(fpath, columns=READ_COLS)
    return {
        "T": t.num_rows,
        "kp_left": _col_to_numpy(t.column("observation.mocap.hand.left.keypoints")).astype(np.float32),
        "kp_right": _col_to_numpy(t.column("observation.mocap.hand.right.keypoints")).astype(np.float32),
        "valid_left": _col_to_numpy(t.column("observation.mocap.hand.left.valid")).astype(bool),
        "valid_right": _col_to_numpy(t.column("observation.mocap.hand.right.valid")).astype(bool),
    }


def _save_glb(viz_record: dict[str, Any]) -> None:
    try:
        import trimesh  # type: ignore
    except Exception as exc:
        print(f"Warning: trimesh unavailable, skip GLB export: {exc}")
        return

    mesh = trimesh.Trimesh(
        vertices=viz_record["pred_verts_local"],
        faces=viz_record["faces"],
        process=False,
    )
    mesh.visual.vertex_colors = [80, 200, 120, 220]

    spheres = []
    for pt in viz_record["markers_local"]:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.004)
        sphere.apply_translation(pt)
        sphere.visual.vertex_colors = [255, 215, 0, 255]
        spheres.append(sphere)

    scene = trimesh.Scene([mesh] + spheres)
    out_path = Path(viz_record["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)


def _choose_viz_indices(num_samples: int, viz_frames: int) -> np.ndarray:
    if viz_frames <= 0 or num_samples <= 0:
        return np.empty((0,), dtype=np.int64)
    count = min(viz_frames, num_samples)
    return np.unique(np.linspace(0, num_samples - 1, num=count, dtype=np.int64))


def mirror_local_points_x(points: torch.Tensor) -> torch.Tensor:
    return points * MIRROR_X_3.to(device=points.device, dtype=points.dtype)


def mirror_local_points_z(points: torch.Tensor) -> torch.Tensor:
    return points * MIRROR_Z_3.to(device=points.device, dtype=points.dtype)


def flip_face_winding(faces: np.ndarray) -> np.ndarray:
    return faces[:, [0, 2, 1]]


def infer_hand(
    model: torch.nn.Module,
    keypoints: np.ndarray,
    valid: np.ndarray,
    stride: int,
    batch_size: int,
    device: torch.device,
    viz_frames: int = 0,
    left_hand_mode: str = "flip_local_z",
    frame_range: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    """
    Args:
        frame_range: Optional (start, end) tuple; if provided, only frames within
            [start, end) are considered for valid sampling. Output pose_full will
            be zero outside this range.
    Returns:
        pose_full: (T, 48) interpolated pose (zero outside frame_range)
        beta_mean: (10,) episode-level mean beta
        sampled_indices: (N,) indices of sampled frames (within valid frames and optionally range)
        viz_samples: selected sampled-frame markers/pose for optional mesh export
    """
    T = keypoints.shape[0]
    frame_valid = valid.any(axis=1) if valid.ndim == 2 else np.ones(T, dtype=bool)

    if frame_range is not None:
        start, end = frame_range
        start = max(0, start)
        end = min(T, end)
        range_mask = np.zeros(T, dtype=bool)
        range_mask[start:end] = True
        valid_indices = np.where(frame_valid & range_mask)[0]
    else:
        valid_indices = np.where(frame_valid)[0]

    if len(valid_indices) == 0:
        # Fallback: sample all frames in range (or whole sequence) with stride
        if frame_range is not None:
            fallback_indices = np.arange(frame_range[0], frame_range[1], stride)
        else:
            fallback_indices = np.arange(0, T, stride)
        sampled_indices = fallback_indices
    else:
        sampled_indices = valid_indices[::stride]

    if len(sampled_indices) == 0:
        sampled_indices = np.array([0], dtype=np.int64)

    sampled_kp = keypoints[sampled_indices]
    all_poses: list[torch.Tensor] = []
    all_betas: list[torch.Tensor] = []
    all_markers_local: list[torch.Tensor] = []
    all_pose_mirrors: list[torch.Tensor] = []

    for i in range(0, len(sampled_kp), batch_size):
        batch = torch.from_numpy(sampled_kp[i : i + batch_size]).float().to(device)
        root = batch[:, 0:1, :].clone()
        batch = batch - root
        batch_local, _ = transform_joints_coordinates_torch(batch)
        if left_hand_mode == "flip_local_x":
            batch_local = mirror_local_points_x(batch_local)
        elif left_hand_mode == "flip_local_z":
            batch_local = mirror_local_points_z(batch_local)
        with torch.no_grad():
            pred_pose_6d, pred_betas = model(batch_local)
        pred_pose_mat = six_d_to_rot_matrix(pred_pose_6d.view(-1, 16, 6))
        pred_pose_ax_mir = matrix_to_axis_angle(pred_pose_mat).view(-1, 48)
        pred_pose_ax = pred_pose_ax_mir.clone()
        pred_pose_ax[:, :3] = 0.0
        pred_pose_ax_mir[:, :3] = 0.0
        all_poses.append(pred_pose_ax.cpu())
        all_betas.append(pred_betas.cpu())
        all_pose_mirrors.append(pred_pose_ax_mir.cpu())
        if viz_frames > 0:
            all_markers_local.append(batch_local.cpu())

    poses_sampled = torch.cat(all_poses, dim=0)
    betas_sampled = torch.cat(all_betas, dim=0)
    beta_mean = betas_sampled.mean(dim=0).numpy().astype(np.float32)

    n = len(sampled_indices)
    if n < 2:
        pose_full = np.zeros((T, 48), dtype=np.float32)
        if n == 1:
            pose_full[:] = poses_sampled[0].numpy()
    else:
        sampled_idx_t = torch.tensor(sampled_indices, dtype=torch.float32, device=device)
        target_t = torch.arange(T, dtype=torch.float32, device=device)
        idx = torch.searchsorted(sampled_idx_t.long(), target_t.long(), right=True) - 1
        idx = idx.clamp(0, n - 2)
        t_low = sampled_idx_t[idx]
        t_high = sampled_idx_t[idx + 1]
        w = (target_t - t_low) / (t_high - t_low + 1e-8)
        pose_full_t = poses_sampled.to(device)[idx] * (1 - w[:, None]) + poses_sampled.to(device)[idx + 1] * w[:, None]
        pose_full = pose_full_t.cpu().numpy().astype(np.float32)

    # Zero out invalid frames and frames outside specified range
    pose_full[~frame_valid] = 0.0
    if frame_range is not None:
        mask_out = np.ones(T, dtype=bool)
        mask_out[frame_range[0]:frame_range[1]] = False
        pose_full[mask_out] = 0.0
    pose_full[:, :3] = 0.0  # global translation always zero

    return {
        "pose_full": pose_full,
        "beta_mean": beta_mean,
        "sampled_indices": sampled_indices,
        "viz_samples": _pack_viz_samples(
            sampled_indices=sampled_indices,
            poses_sampled=poses_sampled,
            poses_mirrored_sampled=torch.cat(all_pose_mirrors, dim=0),
            markers_local_sampled=torch.cat(all_markers_local, dim=0) if all_markers_local else None,
            viz_frames=viz_frames,
        ),
    }


def _pack_viz_samples(
    sampled_indices: np.ndarray,
    poses_sampled: torch.Tensor,
    poses_mirrored_sampled: torch.Tensor,
    markers_local_sampled: torch.Tensor | None,
    viz_frames: int,
) -> list[dict[str, Any]]:
    if viz_frames <= 0 or markers_local_sampled is None or len(sampled_indices) == 0:
        return []

    chosen = _choose_viz_indices(len(sampled_indices), viz_frames)
    markers_np = markers_local_sampled.numpy().astype(np.float32, copy=False)
    poses_np = poses_sampled.numpy().astype(np.float32, copy=False)
    poses_mir_np = poses_mirrored_sampled.numpy().astype(np.float32, copy=False)
    return [
        {
            "frame_idx": int(sampled_indices[j]),
            "markers_local": markers_np[j],
            "pose_axis_angle": poses_np[j],
            "pose_axis_angle_mirrored": poses_mir_np[j],
        }
        for j in chosen
    ]


def build_viz_records(
    episode_stem: str,
    hand_side: str,
    hand_result: dict[str, Any],
    mano_layer: ManoLayer | None,
    device: torch.device,
    faces: np.ndarray | None,
    viz_dir: Path,
    left_hand_mode: str = "flip_local_z",
) -> list[dict[str, Any]]:
    viz_samples = hand_result.get("viz_samples", [])
    if not viz_samples or mano_layer is None or faces is None:
        return []

    pose_key = "pose_axis_angle_mirrored" if hand_side == "left" else "pose_axis_angle"
    poses = torch.from_numpy(np.stack([s[pose_key] for s in viz_samples], axis=0)).to(device)
    betas = torch.from_numpy(
        np.repeat(hand_result["beta_mean"][None, :], len(viz_samples), axis=0)
    ).to(device)

    with torch.no_grad():
        mano_out = mano_layer(poses, betas)
        marker_idx = MARKER_VERT_INDICES.to(device)
        pred_verts_local = transform_hand_coordinates_torch(
            mano_out.verts, mano_out.verts[:, marker_idx, :]
        )
        mirrored_for_display = False
        if hand_side == "left" and left_hand_mode == "flip_local_x":
            pred_verts_local = mirror_local_points_x(pred_verts_local)
            mirrored_for_display = True
        elif hand_side == "left" and left_hand_mode == "flip_local_z":
            pred_verts_local = mirror_local_points_z(pred_verts_local)
            mirrored_for_display = True
    pred_verts_local_np = pred_verts_local.detach().cpu().numpy().astype(np.float32)
    faces_out = flip_face_winding(faces) if mirrored_for_display else faces

    records = []
    for sample, verts_local in zip(viz_samples, pred_verts_local_np, strict=True):
        markers_local = sample["markers_local"]
        if hand_side == "left" and left_hand_mode == "flip_local_x":
            markers_local = mirror_local_points_x(
                torch.from_numpy(markers_local)
            ).cpu().numpy()
        elif hand_side == "left" and left_hand_mode == "flip_local_z":
            markers_local = mirror_local_points_z(
                torch.from_numpy(markers_local)
            ).cpu().numpy()
        records.append(
            {
                "out_path": str(
                    viz_dir / f"{episode_stem}_{hand_side}_frame_{sample['frame_idx']:06d}.glb"
                ),
                "markers_local": markers_local.astype(np.float32, copy=False),
                "pred_verts_local": verts_local,
                "faces": faces_out,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer MANO params for a single EgoEMG episode with optional frame range."
    )
    # Episode selection: either index or explicit name
    parser.add_argument("--episode-idx", type=int, default=None,
                        help="Index of episode in sorted list (0-based).")
    parser.add_argument("--episode-name", type=str, default=None,
                        help="Exact parquet filename, e.g., 'episode_000123.parquet'.")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--stride", type=int, default=100,
                        help="Stride for sampling valid frames (global default).")
    parser.add_argument("--frame-stride", type=int, default=None,
                        help="Stride within the specified frame range (overrides --stride if given).")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Start frame index (inclusive).")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="End frame index (exclusive).")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--viz-frames-per-hand", type=int, default=0,
                        help="Number of visualization meshes to export per hand.")
    parser.add_argument("--output-dir", type=Path, default=MANO_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print("Left-hand strategy: flip_local_z")

    output_dir = args.output_dir
    viz_dir = args.viz_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.viz_frames_per_hand > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # List all parquet files
    episode_files = sorted(
        [
            f
            for f in os.listdir(DATA_DIR)
            if f.endswith(".parquet") and ".bak" not in f and ".mano_bak" not in f
        ]
    )
    print(f"Found {len(episode_files)} episodes in {DATA_DIR}")

    # Determine which single episode to process
    if args.episode_name is not None:
        if args.episode_name not in episode_files:
            raise ValueError(f"Episode '{args.episode_name}' not found.")
        target_fname = args.episode_name
        idx = episode_files.index(target_fname)
        print(f"Selected episode by name: {target_fname} (index {idx})")
    elif args.episode_idx is not None:
        if args.episode_idx < 0 or args.episode_idx >= len(episode_files):
            raise ValueError(f"Episode index {args.episode_idx} out of range (0-{len(episode_files)-1}).")
        target_fname = episode_files[args.episode_idx]
        idx = args.episode_idx
        print(f"Selected episode by index: {target_fname}")
    else:
        # Default to first episode
        target_fname = episode_files[0]
        idx = 0
        print(f"No episode specified, defaulting to first: {target_fname}")

    fpath = DATA_DIR / target_fname
    stem = target_fname.replace(".parquet", "")
    print(f"Processing: {stem}")

    # Load data
    data = read_episode(fpath)
    T = data["T"]
    print(f"Total frames: {T}")

    # Determine frame range and stride
    frame_range = None
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else 0
        end = args.end_frame if args.end_frame is not None else T
        start = max(0, start)
        end = min(T, end)
        if start >= end:
            raise ValueError(f"Invalid frame range: start={start} >= end={end}")
        frame_range = (start, end)
        print(f"Frame range: [{start}, {end})")

    stride = args.frame_stride if args.frame_stride is not None else args.stride
    print(f"Sampling stride: {stride}")

    # Load model and optionally MANO layer
    model = load_model(device)
    mano_layer = None
    faces = None
    if args.viz_frames_per_hand > 0:
        mano_layer = load_mano_layer(device)
        faces = mano_layer.get_mano_closed_faces().detach().cpu().numpy()

    # Inference
    t0 = time.time()
    t_gpu = time.time()

    left_result = infer_hand(
        model,
        data["kp_left"],
        data["valid_left"],
        stride,
        args.batch_size,
        device,
        viz_frames=args.viz_frames_per_hand,
        left_hand_mode="flip_local_z",
        frame_range=frame_range,
    )
    right_result = infer_hand(
        model,
        data["kp_right"],
        data["valid_right"],
        stride,
        args.batch_size,
        device,
        viz_frames=args.viz_frames_per_hand,
        left_hand_mode="none",
        frame_range=frame_range,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_time = time.time() - t_gpu

    # Build visualization records
    viz_records = build_viz_records(
        stem, "left", left_result, mano_layer, device, faces, viz_dir, "flip_local_z"
    )
    viz_records.extend(
        build_viz_records(stem, "right", right_result, mano_layer, device, faces, viz_dir, "none")
    )

    # Save outputs
    np.save(output_dir / f"{stem}_left_pose.npy", left_result["pose_full"])
    np.save(output_dir / f"{stem}_left_beta.npy", left_result["beta_mean"])
    np.save(output_dir / f"{stem}_right_pose.npy", right_result["pose_full"])
    np.save(output_dir / f"{stem}_right_beta.npy", right_result["beta_mean"])
    for rec in viz_records:
        _save_glb(rec)

    elapsed = time.time() - t0
    print(
        f"Done in {elapsed:.1f}s (GPU: {gpu_time:.1f}s).\n"
        f"Left beta: {np.round(left_result['beta_mean'], 2)}\n"
        f"Right beta: {np.round(right_result['beta_mean'], 2)}\n"
        f"Saved to {output_dir}\n"
        f"Visualizations: {len(viz_records)} exported to {viz_dir}"
    )


if __name__ == "__main__":
    main()
