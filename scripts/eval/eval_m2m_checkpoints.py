#!/usr/bin/env python3
"""
Evaluate m2m checkpoints on a small subset of episodes.

Compares left vs right hand aligned marker error for each checkpoint.
Runs on ~5 episodes to quickly identify the best checkpoint.

Usage:
    python scripts/eval/eval_m2m_checkpoints.py --episodes 0 1 2 3 4 --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

WILOR_ROOT = Path("/home/xiziheng/develop/WiLoR")
if str(WILOR_ROOT) not in sys.path:
    sys.path.append(str(WILOR_ROOT))

from manotorch.manolayer import ManoLayer
from markers2mano.geometry import transform_joints_coordinates_torch, matrix_to_axis_angle
from markers2mano.graph_transformer import EfficientGraphTransformer, six_d_to_rot_matrix
from markers2mano.rigid_align import compute_aligned_error

EGOEMG_ROOT = Path("/home/xiziheng/develop/emg2pose/data/EgoEMG")
MANO_ASSETS_ROOT = Path("/home/xiziheng/develop/HandVQVAE/assets/mano")
DATA_DIR = EGOEMG_ROOT / "data" / "chunk-000"

MARKER_VERT_INDICES = torch.tensor(
    [191, 88, 253, 708, 729, 144, 87, 295, 319, 220, 365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673],
    dtype=torch.long,
)

# All candidate checkpoints: (label, checkpoint_path, description)
CHECKPOINTS = [
    ("v43_ep500", WILOR_ROOT / "tb_logs/m2m_pose_shape_run/version_43/checkpoints/last.ckpt", "500ep, hidden=1280, lr=1e-5"),
    ("v51_last_ep34", WILOR_ROOT / "tb_logs/m2m_pose_shape_run/version_51/checkpoints/last.ckpt", "v51 4x, 34ep, lr=1e-5"),
    ("v51_best_ep12", WILOR_ROOT / "tb_logs/m2m_pose_shape_run/version_51/checkpoints/best-model-epoch=12-val/loss_total=0.0002.ckpt", "v51 4x, best@12"),
]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda:"):
        return torch.device(device_arg)
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}")
    return torch.device(device_arg)


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    hidden_dim = hp.get("hidden_dim", 1280)
    num_layers = hp.get("num_layers", 12)
    heads = hp.get("heads", 8)
    ffn_ratio = hp.get("ffn_ratio", 4)

    model = EfficientGraphTransformer(
        num_markers=21,
        beta_dim=10,
        embed_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=heads,
        dropout=0.0,
        ffn_ratio=ffn_ratio,
    )
    state_dict = ckpt["state_dict"]
    # Handle different key prefixes
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        elif k.startswith("net."):
            # Some checkpoints use "net." prefix
            cleaned[k[len("net."):]] = v
        else:
            cleaned[k] = v
    msg = model.load_state_dict(cleaned, strict=False)
    print(f"  Load keys: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    model = model.to(device).eval()
    return model




def read_episode(fpath: Path) -> dict:
    import pyarrow.parquet as pq
    t = pq.read_table(fpath, columns=[
        "observation.mocap.hand.left.keypoints",
        "observation.mocap.hand.right.keypoints",
        "observation.mocap.hand.left.valid",
        "observation.mocap.hand.right.valid",
    ])

    def col_to_numpy(col):
        col = col.combine_chunks()
        col_type = col.type
        if hasattr(col_type, 'value_type') and hasattr(col_type.value_type, 'value_type'):
            flat = col.flatten().flatten().to_numpy(zero_copy_only=False)
            first_val = col[0].as_py()
            if first_val and len(first_val) > 0 and len(first_val[0]) > 0:
                return flat.reshape(-1, len(first_val), len(first_val[0]))
            return np.asarray(col.to_pylist())
        elif hasattr(col_type, 'value_type'):
            flat = col.flatten().to_numpy(zero_copy_only=False)
            first_val = col[0].as_py()
            if first_val and len(first_val) > 0:
                return flat.reshape(-1, len(first_val))
            return flat
        return col.to_numpy(zero_copy_only=False)

    return {
        "kp_left": col_to_numpy(t.column("observation.mocap.hand.left.keypoints")).astype(np.float32),
        "kp_right": col_to_numpy(t.column("observation.mocap.hand.right.keypoints")).astype(np.float32),
        "valid_left": col_to_numpy(t.column("observation.mocap.hand.left.valid")).astype(bool),
        "valid_right": col_to_numpy(t.column("observation.mocap.hand.right.valid")).astype(bool),
    }


def eval_hand(
    model, mano_layer, keypoints, valid, batch_size, device,
    is_left=False,
):
    """Infer MANO params and return mean aligned error in mm."""
    T = keypoints.shape[0]
    frame_valid = valid.any(axis=1) if valid.ndim == 2 else np.ones(T, dtype=bool)
    valid_indices = np.where(frame_valid)[0]
    sampled_indices = valid_indices[::50]
    if len(sampled_indices) == 0:
        sampled_indices = np.arange(0, T, 10)

    sampled_kp = keypoints[sampled_indices]
    all_errors = []

    marker_idx = MARKER_VERT_INDICES.to(device)

    for i in range(0, len(sampled_kp), batch_size):
        batch = torch.from_numpy(sampled_kp[i:i + batch_size]).float().to(device)
        root = batch[:, 0:1, :].clone()
        batch = batch - root
        batch_local, T_w2l = transform_joints_coordinates_torch(batch)

        if is_left:
            batch_local = batch_local * torch.tensor([1.0, 1.0, -1.0], device=batch_local.device)

        with torch.no_grad():
            pred_pose_6d, pred_betas = model(batch_local)

        pred_pose_mat = six_d_to_rot_matrix(pred_pose_6d.view(-1, 16, 6))
        pred_pose_ax = matrix_to_axis_angle(pred_pose_mat).view(-1, 48)
        pred_pose_ax[:, :3] = 0.0

        mano_out = mano_layer(pred_pose_ax, pred_betas)
        pred_markers = mano_out.verts[:, marker_idx, :]

        if is_left:
            pred_markers = pred_markers * torch.tensor([1.0, 1.0, -1.0], device=pred_markers.device)

        gt_local = batch_local
        if is_left:
            gt_local = gt_local * torch.tensor([1.0, 1.0, -1.0], device=gt_local.device)

        for f in range(pred_markers.shape[0]):
            err, _, _ = compute_aligned_error(pred_markers[f], gt_local[f])
            all_errors.append(err.mean().item() * 1000.0)  # mm

    return np.mean(all_errors) if all_errors else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    episode_files = sorted(
        f for f in DATA_DIR.iterdir()
        if f.name.endswith(".parquet") and ".bak" not in f.name and ".mano_bak" not in f.name
        and not f.name.endswith(".parquet.bak") and "video_index" not in f.name
    )
    if args.episodes:
        episode_files = [episode_files[i] for i in args.episodes]
    else:
        episode_files = episode_files[:5]
    print(f"Evaluating on {len(episode_files)} episodes: {[f.stem for f in episode_files]}")

    mano_layer = ManoLayer(
        use_pca=False,
        mano_assets_root=str(MANO_ASSETS_ROOT),
        flat_hand_mean=False,
    ).to(device)

    results = []  # (ckpt_label, left_err, right_err, diff, avg)

    for label, ckpt_path, description in CHECKPOINTS:
        if not ckpt_path.exists():
            print(f"SKIP {label}: not found at {ckpt_path}")
            continue
        print(f"\n=== {label} ({description}) ===")
        print(f"  Checkpoint: {ckpt_path}")

        try:
            model = load_model(ckpt_path, device)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue

        left_errs = []
        right_errs = []

        for ep_file in episode_files:
            data = read_episode(ep_file)

            l_err = eval_hand(model, mano_layer, data["kp_left"], data["valid_left"],
                              args.batch_size, device, is_left=True)
            r_err = eval_hand(model, mano_layer, data["kp_right"], data["valid_right"],
                              args.batch_size, device, is_left=False)
            left_errs.append(l_err)
            right_errs.append(r_err)
            print(f"  {ep_file.stem}: left={l_err:.2f}mm, right={r_err:.2f}mm, diff={abs(l_err-r_err):.2f}mm")

        mean_left = np.mean(left_errs)
        mean_right = np.mean(right_errs)
        mean_both = (mean_left + mean_right) / 2
        print(f"  MEAN: left={mean_left:.2f}mm, right={mean_right:.2f}mm, avg={mean_both:.2f}mm, diff={abs(mean_left-mean_right):.2f}mm")
        results.append((label, description, mean_left, mean_right, mean_both))

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Checkpoint':<20} {'Description':<20} {'Left(mm)':>10} {'Right(mm)':>10} {'Avg(mm)':>10} {'L-R diff':>10}")
    print("-" * 80)
    for label, desc, l, r, avg in sorted(results, key=lambda x: x[4]):
        print(f"{label:<20} {desc:<20} {l:>10.2f} {r:>10.2f} {avg:>10.2f} {abs(l-r):>10.2f}")
    print("=" * 80)

    best = min(results, key=lambda x: x[4])
    print(f"\nBest checkpoint: {best[0]} ({best[1]}) with avg error {best[4]:.2f}mm")


if __name__ == "__main__":
    main()
