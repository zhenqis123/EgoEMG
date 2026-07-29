#!/usr/bin/env python3
"""Compute per-frame local→world Kabsch transforms for all episodes.

For each frame, decodes MANO(pose, beta) to get 21 marker positions in local
frame, then computes Kabsch alignment to world-frame GT mocap keypoints.
Stores R (T, 3, 3) and t (T, 3) per episode per hand.

Usage:
    python scripts/mano/compute_mano_world_transforms.py --device 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
WILOR_ROOT = Path("../WiLoR")
if str(WILOR_ROOT) not in sys.path:
    sys.path.append(str(WILOR_ROOT))

from manotorch.manolayer import ManoLayer

MANO_ASSETS_ROOT = Path("../HandVQVAE/assets/mano")
MEMMAP_DIR = PROJ_ROOT / "data" / "EgoEMG_memmap"
MANO_NPY_DIR = PROJ_ROOT / "data" / "EgoEMG" / "mano" / "chunk-000"

MARKER_VERT_INDICES = torch.tensor(
    [191, 88, 253, 708, 729, 144, 87, 295, 319, 220,
     365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673],
    dtype=torch.long,
)
MIRROR_X_3 = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32)


def kabsch_batch(src: torch.Tensor, tgt: torch.Tensor):
    """Batched Kabsch: src, tgt are (B, N, 3). Returns R (B, 3, 3), t (B, 3)."""
    src_mean = src.mean(dim=1, keepdim=True)
    tgt_mean = tgt.mean(dim=1, keepdim=True)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    H = src_c.transpose(1, 2) @ tgt_c  # (B, 3, 3)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2))  # (B,)
    sign = torch.ones(d.shape[0], 3, device=src.device)
    sign[:, 2] = d.sign()
    R = Vt.transpose(1, 2) @ torch.diag_embed(sign) @ U.transpose(1, 2)  # (B, 3, 3)
    t = tgt_mean.squeeze(1) - (R @ src_mean.squeeze(1).unsqueeze(-1)).squeeze(-1)
    return R, t


def mirror_raw_mano_points_x(points: torch.Tensor) -> torch.Tensor:
    return points * MIRROR_X_3.to(device=points.device, dtype=points.dtype)


def process_episode(
    episode_id: str,
    hand: str,
    mano_layer: ManoLayer,
    gt_keypoints_mm: np.memmap,
    gt_valid_mm: np.memmap,
    ep_start: int,
    ep_end: int,
    device: torch.device,
    batch_size: int = 2000,
):
    pose_path = MANO_NPY_DIR / f"{episode_id}_{hand}_pose.npy"
    beta_path = MANO_NPY_DIR / f"{episode_id}_{hand}_beta.npy"
    if not pose_path.exists():
        return None

    pose_mm = np.load(pose_path, mmap_mode="r")
    beta = np.load(beta_path)
    T = ep_end - ep_start
    assert pose_mm.shape[0] == T, f"pose shape {pose_mm.shape[0]} != episode length {T}"

    marker_idx = MARKER_VERT_INDICES.to(device)
    beta_t = torch.from_numpy(np.array(beta)).float().to(device).unsqueeze(0)

    all_R = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))
    all_t = np.zeros((T, 3), dtype=np.float32)
    valid_computed = np.zeros(T, dtype=bool)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        bs = end - start

        pose_batch = np.array(pose_mm[start:end])
        pose_t = torch.from_numpy(pose_batch).float().to(device)
        beta_batch = beta_t.expand(bs, -1)

        with torch.no_grad():
            mano_out = mano_layer(pose_t, beta_batch)
            pred_markers = mano_out.verts[:, marker_idx, :]
            if hand == "left":
                # EgoEMG left-hand poses are stored in MANO-right semantics. For
                # world alignment we must first mirror the raw MANO geometry into
                # left-hand chirality; a pure-rotation Kabsch fit cannot recover
                # handedness on its own.
                pred_markers = mirror_raw_mano_points_x(pred_markers)

        gt_slice = np.array(gt_keypoints_mm[ep_start + start:ep_start + end],
                            dtype=np.float32)
        gt_batch = torch.from_numpy(gt_slice).float().to(device)

        valid_slice = np.array(gt_valid_mm[ep_start + start:ep_start + end])
        frame_valid = valid_slice.any(axis=1) if valid_slice.ndim == 2 else np.ones(bs, dtype=bool)
        has_data = frame_valid & ~np.isnan(gt_slice[:, 0, 0]) & (np.abs(gt_slice).sum(axis=(1, 2)) > 0)

        valid_idx = np.where(has_data)[0]
        if len(valid_idx) == 0:
            continue

        R_b, t_b = kabsch_batch(pred_markers[valid_idx], gt_batch[valid_idx])
        R_np = R_b.cpu().numpy()
        t_np = t_b.cpu().numpy()
        for i, vi in enumerate(valid_idx):
            all_R[start + vi] = R_np[i]
            all_t[start + vi] = t_np[i]
            valid_computed[start + vi] = True

    if valid_computed.sum() > 0 and valid_computed.sum() < T:
        valid_idx = np.where(valid_computed)[0]
        invalid_idx = np.where(~valid_computed)[0]
        nearest = np.searchsorted(valid_idx, invalid_idx, side="left")
        nearest = np.clip(nearest, 0, len(valid_idx) - 1)
        all_R[invalid_idx] = all_R[valid_idx[nearest]]
        all_t[invalid_idx] = all_t[valid_idx[nearest]]

    return all_R, all_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--hands",
        type=str,
        nargs="+",
        default=["left", "right"],
        choices=["left", "right"],
        help="Hands to recompute. Use '--hands left' to only regenerate left-hand transforms.",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit()
                          else args.device)
    print(f"Device: {device}")

    md = np.load(MEMMAP_DIR / "metadata.npz", allow_pickle=False)
    episode_ids = [v.decode() if isinstance(v, bytes) else str(v)
                   for v in md["episode_id"]]
    ep_starts = md["episode_start_idx"]
    ep_ends = md["episode_end_idx"]

    import json
    with open(MEMMAP_DIR / "manifest.json") as f:
        manifest = json.load(f)

    def open_mm(name):
        info = manifest["fields"][name]
        return np.memmap(
            MEMMAP_DIR / f"{name}.dat",
            dtype=info["dtype"],
            mode="r",
            shape=tuple(info["shape"]),
        )

    mano_layer = ManoLayer(
        use_pca=False,
        mano_assets_root=str(MANO_ASSETS_ROOT),
        flat_hand_mean=False,
    ).to(device)

    total_start = time.time()
    ep_indices = args.episodes if args.episodes else list(range(len(episode_ids)))

    for hand in args.hands:
        kp_key = f"mocap_{hand}_keypoints"
        valid_key = f"mocap_{hand}_valid"
        if kp_key not in manifest["fields"]:
            print(f"Skipping {hand}: {kp_key} not in memmap")
            continue
        gt_kp_mm = open_mm(kp_key)
        gt_valid_mm = open_mm(valid_key)

        for idx in ep_indices:
            ep_id = episode_ids[idx]
            ep_start = int(ep_starts[idx])
            ep_end = int(ep_ends[idx])
            T = ep_end - ep_start

            t0 = time.time()
            result = process_episode(
                ep_id, hand, mano_layer, gt_kp_mm, gt_valid_mm,
                ep_start, ep_end, device, args.batch_size,
            )
            if result is None:
                print(f"[{idx+1}/{len(ep_indices)}] {ep_id} {hand}: SKIP (no pose file)")
                continue

            R, t = result
            R_path = MANO_NPY_DIR / f"{ep_id}_{hand}_world_R.npy"
            t_path = MANO_NPY_DIR / f"{ep_id}_{hand}_world_t.npy"
            np.save(R_path, R)
            np.save(t_path, t)
            elapsed = time.time() - t0
            print(f"[{idx+1}/{len(ep_indices)}] {ep_id} {hand}: "
                  f"{T:,} frames, {elapsed:.1f}s → {R_path.name}, {t_path.name}")

    total = time.time() - total_start
    print(f"\nDone in {total:.1f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
