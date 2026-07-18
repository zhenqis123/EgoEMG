#!/usr/bin/env python3
"""Visualize MANO mesh + GT mocap markers from EgoEmgMemmapDataset.

Loads a dataset sample, decodes MANO parameters into a hand mesh,
and exports a GLB file with the mesh (green) and GT markers (gold spheres).

Usage:
    # Single sample from episode 3, right hand, frame offset 50000
    python scripts/viz/viz_mano_from_dataset.py --episode 3 --hand right --offset 50000

    # Batch: 3 evenly-spaced frames per episode, first 5 episodes
    python scripts/viz/viz_mano_from_dataset.py --episodes 0 1 2 3 4 --num-frames 3

    # All episodes, 1 frame each
    python scripts/viz/viz_mano_from_dataset.py --all --num-frames 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
WILOR_ROOT = Path("/home/xiziheng/develop/WiLoR")
if str(WILOR_ROOT) not in sys.path:
    sys.path.append(str(WILOR_ROOT))

MANO_ASSETS_ROOT = Path("/home/xiziheng/develop/HandVQVAE/assets/mano")
MEMMAP_DIR = PROJ_ROOT / "data" / "EgoEMG_memmap"
MANO_NPY_DIR = PROJ_ROOT / "data" / "EgoEMG" / "mano" / "chunk-000"
DEFAULT_OUT_DIR = PROJ_ROOT / "data" / "EgoEMG" / "mano_viz" / "dataset_samples"

MARKER_VERT_INDICES = torch.tensor(
    [191, 88, 253, 708, 729, 144, 87, 295, 319, 220,
     365, 407, 445, 183, 477, 518, 556, 83, 589, 635, 673],
    dtype=torch.long,
)
MIRROR_X_3 = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


def make_dataset(hand: str, window_length: int = 1000):
    from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    return EgoEmgMemmapDataset(
        memmap_dir=MEMMAP_DIR,
        window_length=window_length,
        stride=window_length,
        modalities=["emg", "joint_angles", "mocap_hands", "mano"],
        target_hand=hand,
        emg_field_preference="filtered",
        emg_layout="target_hand",
        norm_mode=None,
        dataset_name="egoemg",
        mano_npy_dir=MANO_NPY_DIR,
    )


def decode_mano(
    pose: np.ndarray,
    beta: np.ndarray,
    device: torch.device,
    hand: str,
):
    """Decode EgoEMG MANO params → vertices and marker positions."""
    from manotorch.manolayer import ManoLayer

    mano_layer = ManoLayer(
        use_pca=False,
        mano_assets_root=str(MANO_ASSETS_ROOT),
        flat_hand_mean=False,
    ).to(device)

    pose_t = torch.from_numpy(np.array(pose)).float().to(device)
    beta_t = torch.from_numpy(np.array(beta)).float().to(device)
    if pose_t.ndim == 1:
        pose_t = pose_t.unsqueeze(0)
    if beta_t.ndim == 1:
        beta_t = beta_t.unsqueeze(0)

    with torch.no_grad():
        out = mano_layer(pose_t, beta_t)
    verts = out.verts[0].cpu().numpy()
    faces = mano_layer.th_faces.cpu().numpy()
    markers = out.verts[0, MARKER_VERT_INDICES.to(device)].cpu().numpy()
    if hand == "left":
        verts = verts * MIRROR_X_3.astype(verts.dtype, copy=False)
        markers = markers * MIRROR_X_3.astype(markers.dtype, copy=False)
        faces = faces[:, [0, 2, 1]]
    return verts, faces, markers


def save_glb(out_path: Path, verts: np.ndarray, faces: np.ndarray,
             gt_markers: np.ndarray | None = None,
             pred_markers: np.ndarray | None = None):
    """Export mesh + marker spheres as GLB."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = [80, 200, 120, 220]
    parts = [mesh]

    if gt_markers is not None:
        for pt in gt_markers:
            s = trimesh.creation.icosphere(subdivisions=2, radius=0.004)
            s.apply_translation(pt)
            s.visual.vertex_colors = [255, 215, 0, 255]
            parts.append(s)

    if pred_markers is not None:
        for pt in pred_markers:
            s = trimesh.creation.icosphere(subdivisions=2, radius=0.003)
            s.apply_translation(pt)
            s.visual.vertex_colors = [65, 105, 225, 200]
            parts.append(s)

    scene = trimesh.Scene(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))


def find_window_indices(ds, ep_idx: int) -> list[int]:
    """Return dataset window indices that belong to a given episode."""
    block_mask = ds._block_episode_idx == ep_idx
    block_ids = np.where(block_mask)[0]
    indices = []
    for bi in block_ids:
        w_start = int(ds._block_cumsum[bi])
        w_end = int(ds._block_cumsum[bi + 1])
        indices.extend(range(w_start, w_end))
    return indices


def process_sample(ds, sample_idx: int, device: torch.device,
                   out_dir: Path, hand: str) -> str | None:
    sample = ds[sample_idx]
    if "mano_pose" not in sample or "mano_beta" not in sample:
        return None

    pose = sample["mano_pose"]
    beta = sample["mano_beta"]
    mid = pose.shape[0] // 2
    verts, faces, pred_markers = decode_mano(pose[mid], beta, device, hand)

    gt_markers = None
    if "mocap_keypoints" in sample:
        kp = sample["mocap_keypoints"]
        if kp.ndim == 3 and kp.shape[0] > mid:
            gt_markers = kp[mid].astype(np.float32)

    if gt_markers is not None and "mano_world_R" in sample and "mano_world_t" in sample:
        R = torch.from_numpy(sample["mano_world_R"][mid].copy()).float()
        t = torch.from_numpy(sample["mano_world_t"][mid].copy()).float()
        verts = (torch.from_numpy(verts).float() @ R.T + t).numpy()
        pred_markers = (torch.from_numpy(pred_markers).float() @ R.T + t).numpy()
    elif gt_markers is not None:
        src = torch.from_numpy(pred_markers).float()
        tgt = torch.from_numpy(gt_markers).float()
        src_c = src - src.mean(dim=0, keepdim=True)
        tgt_c = tgt - tgt.mean(dim=0, keepdim=True)
        H = src_c.T @ tgt_c
        U, S, Vt = torch.linalg.svd(H)
        d = torch.det(Vt.T @ U.T)
        sign = torch.diag(torch.tensor([1.0, 1.0, d.sign()]))
        R = Vt.T @ sign @ U.T
        t = tgt.mean(dim=0) - (R @ src.mean(dim=0))
        verts = (torch.from_numpy(verts).float() @ R.T + t).numpy()
        pred_markers = (torch.from_numpy(pred_markers).float() @ R.T + t).numpy()

    bi = int(np.searchsorted(ds._block_cumsum, sample_idx, side="right") - 1)
    ep_idx = int(ds._block_episode_idx[bi])
    block_start = int(ds._block_start[bi])
    rel = int(sample_idx - ds._block_cumsum[bi])
    start = block_start + rel * ds.stride
    ep_id = ds._episode_id[ep_idx]
    frame = start + mid
    fname = f"{ep_id}_{hand}_frame{frame:07d}.glb"
    out_path = out_dir / fname
    save_glb(out_path, verts, faces, gt_markers, pred_markers)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, default=None,
                        help="Single episode index")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="List of episode indices")
    parser.add_argument("--all", action="store_true",
                        help="Process all episodes")
    parser.add_argument("--hand", type=str, default="right",
                        choices=["left", "right"])
    parser.add_argument("--offset", type=int, default=None,
                        help="Frame offset within episode (single-sample mode)")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Number of evenly-spaced frames per episode (batch mode)")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu"
                          else f"cuda:{args.device}" if args.device.isdigit()
                          else args.device)

    ds = make_dataset(args.hand, args.window)
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

    total_saved = 0
    for ep_idx in ep_list:
        win_indices = find_window_indices(ds, ep_idx)
        if not win_indices:
            print(f"  episode {ep_idx}: no windows, skipping")
            continue

        if args.offset is not None and len(ep_list) == 1:
            target_win = args.offset // args.window
            idx = win_indices[min(target_win, len(win_indices) - 1)]
            path = process_sample(ds, idx, device, args.out_dir, args.hand)
            if path:
                print(f"  Saved: {path}")
                total_saved += 1
        else:
            n = min(args.num_frames, len(win_indices))
            chosen = np.linspace(0, len(win_indices) - 1, n, dtype=int)
            for c in chosen:
                path = process_sample(ds, win_indices[c], device,
                                      args.out_dir, args.hand)
                if path:
                    print(f"  Saved: {path}")
                    total_saved += 1

    print(f"\nDone. {total_saved} GLB files saved to {args.out_dir}")


if __name__ == "__main__":
    main()
