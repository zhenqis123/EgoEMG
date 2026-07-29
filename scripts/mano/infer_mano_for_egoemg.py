"""
Infer MANO pose & beta for EgoEMG keypoints.

This version separates CPU I/O and GPU inference more aggressively:
- one reader process loads parquet episodes ahead of time;
- the main process only performs GPU inference;
- one writer process persists numpy outputs and optional GLB visualizations.

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
    python scripts/mano/infer_mano_for_egoemg.py --device 0 --episodes 0 --viz-frames-per-hand 1
    python scripts/mano/infer_mano_for_egoemg.py --device 0 --max-episodes 2 --stride 500
    python scripts/mano/infer_mano_for_egoemg.py --device 0 --no-save  # compute errors only, no output
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

WILOR_ROOT = Path("../WiLoR")
if str(WILOR_ROOT) not in sys.path:
    sys.path.append(str(WILOR_ROOT))

from manotorch.manolayer import ManoLayer
from markers2mano.geometry import (
    matrix_to_axis_angle,
    transform_joints_coordinates_torch,
)
from markers2mano.graph_transformer import EfficientGraphTransformer, six_d_to_rot_matrix
from markers2mano.rigid_align import (
    compute_aligned_error,
    compute_aligned_error_batched,
)

EGOEMG_ROOT = Path("./data/EgoEMG")
CKPT_PATH = Path(
    "../WiLoR/tb_logs/m2m_pose_shape_run/version_51/checkpoints/best-model-epoch=12-val/loss_total=0.0002.ckpt"
)
MANO_ASSETS_ROOT = Path("../HandVQVAE/assets/mano")
DATA_DIR = EGOEMG_ROOT / "data" / "chunk-000"
# Output dirs are overridden by --output-dir / --viz-dir args
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


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    hidden_dim = hp.get("hidden_dim", 1280)
    num_layers = hp.get("num_layers", 12)
    heads = hp.get("heads", 8)

    state_dict = ckpt["state_dict"]
    cleaned = {
        k[len("model.") :] if k.startswith("model.") else k: v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    if any(key.startswith("pose_head.") for key in cleaned):
        # v51 was trained with the archived global pose-head architecture.
        # Loading it into the newer per-joint-head class with strict=False
        # silently leaves an identity-initialized pose head and produces all
        # zero axis-angle labels.
        from markers2mano.archive.graph_transformer_4x import (
            EfficientGraphTransformer4x,
        )

        model = EfficientGraphTransformer4x(
            num_markers=21,
            beta_dim=10,
            embed_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=heads,
            dropout=0.0,
            ffn_ratio=hp.get("ffn_ratio", 4),
        )
        architecture = "EfficientGraphTransformer4x"
    else:
        model = EfficientGraphTransformer(
            num_markers=21,
            beta_dim=10,
            embed_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=heads,
            dropout=0.0,
            ffn_ratio=hp.get("ffn_ratio", 4),
        )
        architecture = "EfficientGraphTransformer"
    incompatible = model.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "markers2mano checkpoint architecture mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device).eval()
    print(
        f"Model loaded: architecture={architecture}, dim={hidden_dim}, "
        f"layers={num_layers}, heads={heads}"
    )
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


def reader_worker(
    episode_files: list[str],
    indices: list[int],
    out_queue: mp.Queue,
) -> None:
    try:
        for idx in indices:
            fname = episode_files[idx]
            fpath = DATA_DIR / fname
            stem = fname.replace(".parquet", "")
            out_queue.put(
                {
                    "idx": idx,
                    "fname": fname,
                    "stem": stem,
                    "data": read_episode(fpath),
                }
            )
    finally:
        out_queue.put(None)


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


def writer_worker(in_queue: mp.Queue) -> None:
    while True:
        task = in_queue.get()
        if task is None:
            return

        output_dir = Path(task["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'{task["stem"]}_left_pose.npy', task["left_pose"])
        np.save(output_dir / f'{task["stem"]}_left_beta.npy', task["left_beta"])
        np.save(output_dir / f'{task["stem"]}_left_trans.npy', task["left_trans"])
        np.save(output_dir / f'{task["stem"]}_right_pose.npy', task["right_pose"])
        np.save(output_dir / f'{task["stem"]}_right_beta.npy', task["right_beta"])
        np.save(output_dir / f'{task["stem"]}_right_trans.npy', task["right_trans"])

        for viz_record in task.get("viz_records", []):
            try:
                _save_glb(viz_record)
            except Exception as exc:
                print(f'Warning: failed to save {viz_record["out_path"]}: {exc}')


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
    mano_layer: ManoLayer | None,
    keypoints: np.ndarray,
    valid: np.ndarray,
    stride: int,
    batch_size: int,
    device: torch.device,
    viz_frames: int = 0,
    left_hand_mode: str = "flip_local_z",
) -> dict[str, Any]:
    """
    Returns:
        pose_full: (T, 48) interpolated pose
        beta_mean: (10,) episode-level mean beta
        trans_mean: (3,) episode-level mean rigid-alignment translation
        aligned_error_mm: float — mean marker residual after Kabsch alignment
        sampled_indices: (N,) indices of sampled frames
        viz_samples: selected sampled-frame markers/pose for optional mesh export
    """
    T = keypoints.shape[0]
    frame_valid = valid.any(axis=1) if valid.ndim == 2 else np.ones(T, dtype=bool)
    valid_indices = np.where(frame_valid)[0]
    if len(valid_indices) == 0:
        return {
            "pose_full": np.zeros((T, 48), dtype=np.float32),
            "beta_mean": np.zeros(10, dtype=np.float32),
            "trans_mean": np.zeros(3, dtype=np.float32),
            "aligned_error_mm": float("nan"),
            "sampled_indices": np.empty((0,), dtype=np.int64),
            "viz_samples": [],
        }
    sampled_indices = valid_indices[::stride]

    sampled_kp = keypoints[sampled_indices]
    all_poses: list[torch.Tensor] = []
    all_betas: list[torch.Tensor] = []
    all_markers_local: list[torch.Tensor] = []

    # Per-frame rigid alignment accumulators
    all_trans: list[torch.Tensor] = []
    all_aligned_errors: list[torch.Tensor] = []

    marker_idx = MARKER_VERT_INDICES.to(device)

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
        pred_pose_ax = matrix_to_axis_angle(pred_pose_mat).view(-1, 48)
        pred_pose_ax[:, :3] = 0.0
        all_poses.append(pred_pose_ax.cpu())
        all_betas.append(pred_betas.cpu())

        # --- Rigid alignment: predicted MANO markers vs GT local keypoints ---
        # Both pred_surface_markers and batch_local are in the same coordinate
        # frame (right-hand canonical for right hand; flip_local_z-mirrored for
        # left hand), so Kabsch works correctly for both hands.
        if mano_layer is not None:
            with torch.no_grad():
                mano_out = mano_layer(pred_pose_ax, pred_betas)
                pred_surface_markers = mano_out.verts[:, marker_idx, :]  # (B, 21, 3)
            errors, _, translations = compute_aligned_error_batched(
                pred_surface_markers, batch_local
            )
            all_trans.extend(translations.cpu().unbind(0))
            all_aligned_errors.extend(errors.cpu().unbind(0))

        if viz_frames > 0:
            all_markers_local.append(batch_local.cpu())

    poses_sampled = torch.cat(all_poses, dim=0)
    betas_sampled = torch.cat(all_betas, dim=0)
    beta_mean = betas_sampled.mean(dim=0).numpy().astype(np.float32)

    # Episode-level rigid alignment summary
    if all_trans:
        trans_mean = torch.stack(all_trans).mean(dim=0).numpy().astype(np.float32)
        aligned_error_mm = (torch.cat(all_aligned_errors).mean().item() * 1000.0)
    else:
        trans_mean = np.zeros(3, dtype=np.float32)
        aligned_error_mm = 0.0

    n = len(sampled_indices)
    if n < 2:
        pose_full = np.zeros((T, 48), dtype=np.float32)
        if n == 1:
            pose_full[:] = poses_sampled[0].numpy()
        return {
            "pose_full": pose_full,
            "beta_mean": beta_mean,
            "trans_mean": trans_mean,
            "aligned_error_mm": aligned_error_mm,
            "sampled_indices": sampled_indices,
            "viz_samples": _pack_viz_samples(
                sampled_indices=sampled_indices,
                poses_sampled=poses_sampled,
                markers_local_sampled=torch.cat(all_markers_local, dim=0) if all_markers_local else None,
                viz_frames=viz_frames,
            ),
        }

    sampled_idx_t = torch.tensor(sampled_indices, dtype=torch.float32, device=device)
    target_t = torch.arange(T, dtype=torch.float32, device=device)
    poses_t = poses_sampled.to(device)

    idx = torch.searchsorted(sampled_idx_t.long(), target_t.long(), right=True) - 1
    idx = idx.clamp(0, n - 2)

    t_low = sampled_idx_t[idx]
    t_high = sampled_idx_t[idx + 1]
    w = (target_t - t_low) / (t_high - t_low + 1e-8)

    pose_full_t = poses_t[idx] * (1 - w[:, None]) + poses_t[idx + 1] * w[:, None]
    pose_full = pose_full_t.cpu().numpy().astype(np.float32)

    pose_full[~frame_valid] = 0.0
    pose_full[:, :3] = 0.0

    return {
        "pose_full": pose_full,
        "beta_mean": beta_mean,
        "trans_mean": trans_mean,
        "aligned_error_mm": aligned_error_mm,
        "sampled_indices": sampled_indices,
        "viz_samples": _pack_viz_samples(
            sampled_indices=sampled_indices,
            poses_sampled=poses_sampled,
            markers_local_sampled=torch.cat(all_markers_local, dim=0) if all_markers_local else None,
            viz_frames=viz_frames,
        ),
    }


def _pack_viz_samples(
    sampled_indices: np.ndarray,
    poses_sampled: torch.Tensor,
    markers_local_sampled: torch.Tensor | None,
    viz_frames: int,
) -> list[dict[str, Any]]:
    if viz_frames <= 0 or markers_local_sampled is None or len(sampled_indices) == 0:
        return []

    chosen = _choose_viz_indices(len(sampled_indices), viz_frames)
    markers_np = markers_local_sampled.numpy().astype(np.float32, copy=False)
    poses_np = poses_sampled.numpy().astype(np.float32, copy=False)
    return [
        {
            "frame_idx": int(sampled_indices[j]),
            "markers_local": markers_np[j],
            "pose_axis_angle": poses_np[j],
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
    import trimesh  # type: ignore

    viz_samples = hand_result.get("viz_samples", [])
    if not viz_samples or mano_layer is None or faces is None:
        return []

    poses = torch.from_numpy(np.stack([s["pose_axis_angle"] for s in viz_samples], axis=0)).to(device)
    betas = torch.from_numpy(
        np.repeat(hand_result["beta_mean"][None, :], len(viz_samples), axis=0)
    ).to(device)

    with torch.no_grad():
        mano_out = mano_layer(poses, betas)
        marker_idx = MARKER_VERT_INDICES.to(device)
        pred_verts = mano_out.verts
        pred_markers = pred_verts[:, marker_idx, :]

    records = []
    for sample, verts, markers_pred in zip(
        viz_samples,
        pred_verts.detach().cpu(),
        pred_markers.detach().cpu(),
        strict=True,
    ):
        gt_local = torch.from_numpy(sample["markers_local"])
        # Visualization is auxiliary to label generation.  A pathological
        # prediction (or source frame) must not abort a long full-dataset run.
        if not (
            torch.isfinite(verts).all()
            and torch.isfinite(markers_pred).all()
            and torch.isfinite(gt_local).all()
        ):
            continue
        _, R, t = compute_aligned_error(markers_pred, gt_local)
        verts_aligned = (verts @ R.T + t).numpy().astype(np.float32)

        records.append(
            {
                "out_path": str(
                    viz_dir / f"{episode_stem}_{hand_side}_frame_{sample['frame_idx']:06d}.glb"
                ),
                "markers_local": sample["markers_local"].astype(np.float32, copy=False),
                "pred_verts_local": verts_aligned,
                "faces": faces,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer MANO params for EgoEMG (left hand uses fixed flip_local_z strategy)"
    )
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH,
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--viz-frames-per-hand", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=MANO_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    parser.add_argument("--no-save", action="store_true",
                        help="Compute alignment errors only, do not save any outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print("Left-hand strategy: flip_local_z")

    output_dir = args.output_dir
    viz_dir = args.viz_dir
    save_outputs = not args.no_save

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    if args.viz_frames_per_hand > 0:
        if args.no_save:
            print("Warning: --viz-frames-per-hand > 0 but --no-save is set, ignoring viz.")
        else:
            viz_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(
        [
            f
            for f in os.listdir(DATA_DIR)
            if f.endswith(".parquet") and ".bak" not in f and ".mano_bak" not in f
            and "video_index" not in f
        ]
    )
    print(f"Found {len(episode_files)} episodes")

    if args.episodes is not None:
        indices = list(args.episodes)
    else:
        indices = list(range(len(episode_files)))
    if args.max_episodes is not None:
        indices = indices[: args.max_episodes]
    print(f"Processing {len(indices)} episodes")
    if args.no_save:
        print("Mode: compute-only (no outputs saved)")

    model = load_model(args.ckpt, device)
    mano_layer = load_mano_layer(device)
    faces = None
    if args.viz_frames_per_hand > 0 and save_outputs:
        faces = mano_layer.get_mano_closed_faces().detach().cpu().numpy()

    ctx = mp.get_context("spawn")
    read_queue: mp.Queue = ctx.Queue(maxsize=max(1, args.prefetch))
    if save_outputs:
        write_queue: mp.Queue = ctx.Queue(maxsize=max(1, args.prefetch))

    reader = ctx.Process(target=reader_worker, args=(episode_files, indices, read_queue), daemon=True)
    reader.start()
    writer = None
    if save_outputs:
        writer = ctx.Process(target=writer_worker, args=(write_queue,), daemon=True)
        writer.start()

    t0 = time.time()
    gpu_total = 0.0
    left_errors: list[float] = []
    right_errors: list[float] = []
    left_betas: list[np.ndarray] = []
    right_betas: list[np.ndarray] = []

    try:
        for i in range(len(indices)):
            item = read_queue.get()
            if item is None:
                raise RuntimeError("Reader exited before producing all episodes.")

            stem = item["stem"]
            data = item["data"]
            T = data["T"]
            n_inf = max(1, T // args.stride)

            t_ep = time.time()
            t_gpu = time.time()
            left_result = infer_hand(
                model,
                mano_layer,
                data["kp_left"],
                data["valid_left"],
                args.stride,
                args.batch_size,
                device,
                viz_frames=0 if args.no_save else args.viz_frames_per_hand,
                left_hand_mode="flip_local_z",
            )
            right_result = infer_hand(
                model,
                mano_layer,
                data["kp_right"],
                data["valid_right"],
                args.stride,
                args.batch_size,
                device,
                viz_frames=0 if args.no_save else args.viz_frames_per_hand,
                left_hand_mode="none",
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gpu_time = time.time() - t_gpu
            gpu_total += gpu_time

            viz_records: list[dict[str, Any]] = []
            if save_outputs:
                viz_records = build_viz_records(
                    stem, "left", left_result, mano_layer, device, faces, viz_dir, "flip_local_z"
                )
                viz_records.extend(
                    build_viz_records(stem, "right", right_result, mano_layer, device, faces, viz_dir, "none")
                )

            if save_outputs:
                write_queue.put(
                    {
                        "stem": stem,
                        "output_dir": str(output_dir),
                        "left_pose": left_result["pose_full"],
                        "left_beta": left_result["beta_mean"],
                        "left_trans": left_result["trans_mean"],
                        "right_pose": right_result["pose_full"],
                        "right_beta": right_result["beta_mean"],
                        "right_trans": right_result["trans_mean"],
                        "viz_records": viz_records,
                    }
                )

            left_errors.append(left_result["aligned_error_mm"])
            right_errors.append(right_result["aligned_error_mm"])
            left_betas.append(left_result["beta_mean"])
            right_betas.append(right_result["beta_mean"])

            ep_time = time.time() - t_ep
            print(
                f"[{i + 1}/{len(indices)}] {stem}: {T:,}fr -> {n_inf:,}inf"
                f" | GPU={gpu_time:.1f}s total={ep_time:.1f}s"
                f" | Lb={np.round(left_result['beta_mean'], 2)}"
                f" Rb={np.round(right_result['beta_mean'], 2)}"
                f" | Lerr={left_result['aligned_error_mm']:.1f}mm"
                f" Rerr={right_result['aligned_error_mm']:.1f}mm"
                f" | Lt={np.round(left_result['trans_mean'], 3)}"
                f" Rt={np.round(right_result['trans_mean'], 3)}"
                + (f" | viz={len(viz_records)}" if save_outputs else "")
            )

        tail = read_queue.get(timeout=5)
        if tail is not None:
            raise RuntimeError("Reader queue protocol error: missing sentinel.")
    except queue.Empty:
        raise RuntimeError("Timed out waiting for reader sentinel.") from None
    finally:
        if save_outputs:
            write_queue.put(None)
        reader.join(timeout=10)
        if writer is not None:
            writer.join(timeout=60)
            if writer.is_alive():
                writer.terminate()
        if reader.is_alive():
            reader.terminate()

    elapsed = time.time() - t0
    print(
        f"\nDone! {len(indices)} episodes in {elapsed:.1f}s"
        f" (GPU: {gpu_total:.1f}s, non-GPU: {elapsed - gpu_total:.1f}s)"
    )

    # Aggregate error statistics
    le = np.array(left_errors)
    re = np.array(right_errors)
    lb = np.stack(left_betas)
    rb = np.stack(right_betas)
    combined = (le + re) / 2.0
    print("\n=== Aggregate Alignment Error (mm) ===")
    print(f"{'Stat':<10} {'Left':>10} {'Right':>10} {'Combined':>10}")
    for label, lv, rv, cv in [
        ("count", len(le), len(re), len(combined)),
        ("mean", np.mean(le), np.mean(re), np.mean(combined)),
        ("median", np.median(le), np.median(re), np.median(combined)),
        ("min", np.min(le), np.min(re), np.min(combined)),
        ("max", np.max(le), np.max(re), np.max(combined)),
    ]:
        print(f"{label:<10} {lv:>10.1f} {rv:>10.1f} {cv:>10.1f}")
    print(f"\nMean beta across episodes:")
    print(f"  Left : {np.round(lb.mean(axis=0), 3)}")
    print(f"  Right: {np.round(rb.mean(axis=0), 3)}")


if __name__ == "__main__":
    main()
