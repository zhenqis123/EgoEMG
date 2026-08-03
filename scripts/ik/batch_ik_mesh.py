#!/usr/bin/env python3
"""
Batch IK: optimize UmeTrack 20D angles for every frame in a memmap dataset.
Parallelized across multiple GPUs via multiprocessing. Processes both hands
sequentially. Subsamples frames, solves IK on anchors only, then interpolates.

Alignment (hardcoded, session-independent calibration):
    MANO mesh --[flip X]--> [scale] --> [rotate(opt_global_orient)] --> [translate] --> UmeTrack space

Optimization: L-BFGS over raw_angles + opt global_orient + opt translation,
sigmoid-constrained joint angles. Loss: chamfer and/or landmark L2.

Usage:
    python scripts/ik/batch_ik_mesh.py --gpus 0 --hand right --loss-type landmark --max-iter 200
    python scripts/ik/batch_ik_mesh.py --gpus 0,1,2,3 --loss-type chamfer --batch-size 620
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

MANOTORCH_ROOT = Path("../manotorch")
if str(MANOTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MANOTORCH_ROOT))
from manotorch.manolayer import ManoLayer

MANO_ASSETS_ROOT = Path("../HandVQVAE/assets/mano")

# ── Alignment constants (MANO rest → UmeTrack rest, flat_hand_mean=True) ──
# Session-independent MANO-rest to UmeTrack-rest calibration constants.
ALIGN_SCALE = 1.0843137502670288
ALIGN_TRANS = np.array([106.72334, -11.8804455, -4.48328], dtype=np.float32)

# Negate X to convert MANO-right into UmeTrack's left-hand frame.
FLIP_MATRIX = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

# Landmark mapping: MANO joint index → UmeTrack landmark index.
MANO_TO_UMETRACK = {
    0: 5, 2: 6, 3: 7, 4: 0,
    5: 8, 6: 9, 7: 10, 8: 1,
    9: 11, 10: 12, 11: 13, 12: 2,
    13: 14, 14: 15, 15: 16, 16: 3,
    17: 17, 18: 18, 19: 19, 20: 4,
}
MANO_IDX = sorted(MANO_TO_UMETRACK.keys())
UMETRACK_IDX = [MANO_TO_UMETRACK[m] for m in MANO_IDX]
_MANO_IDX = torch.tensor(MANO_IDX, dtype=torch.long)
_UMETRACK_IDX = torch.tensor(UMETRACK_IDX, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AnchorPoseDataset(torch.utils.data.Dataset):
    """Yields anchor (frame_index, pose) pairs within [start, end) at `step` intervals."""

    def __init__(self, pose_memmap, start_frame, end_frame, step,
                 valid_mask=None):
        self.data = pose_memmap
        first = ((start_frame + step) // step) * step
        last = ((end_frame - 1) // step) * step

        indices = [start_frame]
        a = first
        while a <= last and a < end_frame:
            if a != indices[-1]:
                indices.append(a)
            a += step
        if indices[-1] < end_frame - 1:
            indices.append(end_frame - 1)
        elif len(indices) > 1 and indices[-1] > end_frame - 1:
            indices[-1] = end_frame - 1

        # Filter out frames with invalid (NaN) MANO poses.
        if valid_mask is not None:
            indices = [i for i in indices if valid_mask[i]]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fi = self.indices[idx]
        return fi, self.data[fi].copy().astype(np.float32)


def collate_poses(batch):
    indices = [b[0] for b in batch]
    poses = np.stack([b[1] for b in batch])
    return torch.tensor(indices, dtype=torch.long), torch.from_numpy(poses)


# ═══════════════════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def axisangle_to_rotmat_batch(aa):
    """Axis-angle (B,3) → rotation matrix (B,3,3)."""
    B = aa.shape[0]
    K = torch.zeros(B, 3, 3, device=aa.device)
    K[:, 0, 1] = -aa[:, 2]; K[:, 0, 2] = aa[:, 1]
    K[:, 1, 0] = aa[:, 2]; K[:, 1, 2] = -aa[:, 0]
    K[:, 2, 0] = -aa[:, 1]; K[:, 2, 1] = aa[:, 0]
    return torch.linalg.matrix_exp(K)


def _uniform_idx(n_verts, n_sample):
    return torch.linspace(0, n_verts - 1, n_sample, dtype=torch.long).unique()


# ═══════════════════════════════════════════════════════════════════════════════
# FK helpers
# ═══════════════════════════════════════════════════════════════════════════════

def mano_fk_batch(mano_layer, pose, beta):
    """MANO FK. Returns (joints (B,21,3), verts (B,778,3)) in mm."""
    with torch.no_grad():
        out = mano_layer(pose, beta)
    return out.joints * 1000.0, out.verts * 1000.0


def umetrack_fk_batch(hand_model, angles_22, device, *,
                      return_mesh=True, return_landmarks=False):
    """Batched UmeTrack FK. Returns mesh (B,788,3) and/or landmarks (B,21,3)."""
    from egoemg.kinematics import apply_to_hand_model, broadcast_hand_model_to
    from egoemg.UmeTrack.lib.common.hand_skinning import (
        _get_skinned_vertices, _hand_skinning_transform, _lbs, skin_landmarks,
    )
    B = angles_22.shape[0]
    hm = broadcast_hand_model_to(hand_model, (B,))
    hm = apply_to_hand_model(hm, lambda t: t.float())
    wrist_tf = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)

    lm = skin_landmarks(hm, angles_22[:, :20], wrist_tf) if return_landmarks else None
    mesh = None
    if return_mesh:
        skin_xfs = _hand_skinning_transform(
            hm.joint_rotation_axes.reshape(B, -1, 3),
            hm.joint_rest_positions.reshape(B, -1, 3),
            angles_22, wrist_tf,
        )
        w = hm.dense_bone_weights.reshape(B, -1, 17)
        mr = hm.mesh_vertices.reshape(B, -1, 3)
        v = _get_skinned_vertices(mr, w)
        mesh = _lbs(skin_xfs, v)[..., :3]

    if return_mesh and return_landmarks:
        return mesh, lm
    if return_mesh:
        return mesh
    if return_landmarks:
        return lm
    raise ValueError("Must request at least one of mesh or landmarks")


# ═══════════════════════════════════════════════════════════════════════════════
# Loss helpers
# ═══════════════════════════════════════════════════════════════════════════════

def chamfer_sym_sampled(a, b, a_idx, b_idx):
    def _asym(src, dst, src_idx, dst_idx):
        s = src[:, src_idx, :]
        d = dst[:, dst_idx, :]
        return (s.unsqueeze(2) - d.unsqueeze(1)).pow(2).sum(-1).min(dim=2).values.mean(dim=1).mean()
    return _asym(a, b, a_idx, b_idx) + _asym(b, a, b_idx, a_idx)


def landmark_l2_batch(pred_lm, target_lm):
    return ((pred_lm - target_lm) ** 2).sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Batch optimizer
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_batch(
    hand_model, mano_layer,
    pose_batch,           # (B, 48) — global_orient already zeroed
    flip_t,
    angle_min, angle_max,
    max_iter,
    opt_global_orient, opt_trans,
    loss_type,
    chamfer_weight, landmark_weight,
    mano_vidx, ut_vidx,
    device,
):
    """L-BFGS: fit UmeTrack angles + optional rotation/translation to MANO targets."""
    B = pose_batch.shape[0]
    angle_range = angle_max - angle_min
    use_ch = loss_type in ("chamfer", "both")
    use_lm = loss_type in ("landmark", "both")

    # Optimizable parameters.
    raw_angles = torch.zeros(B, 20, device=device, requires_grad=True)
    opt_params = [raw_angles]

    if opt_global_orient:
        global_orient_raw = torch.zeros(B, 3, device=device, requires_grad=True)
        opt_params.append(global_orient_raw)
    else:
        global_orient_raw = torch.zeros(B, 3, device=device)

    if opt_trans:
        trans_raw = torch.zeros(B, 3, device=device, requires_grad=True)
        opt_params.append(trans_raw)
    else:
        trans_raw = torch.zeros(B, 3, device=device)

    trans_init = torch.tensor(ALIGN_TRANS.tolist(), dtype=torch.float32, device=device)

    def _R():
        return axisangle_to_rotmat_batch(global_orient_raw)

    def _trans():
        return trans_init + 10.0 * trans_raw

    def _transform(pts):
        """Flip X → scale → rotate(opt) → translate."""
        return ALIGN_SCALE * (pts @ flip_t.T) @ _R().transpose(-1, -2) + _trans().unsqueeze(1)

    # Target MANO geometry (fixed).
    beta = torch.zeros(B, 10, device=device)
    mano_j, mano_v = mano_fk_batch(mano_layer, pose_batch, beta)

    mano_idx = _MANO_IDX.to(device)
    utrack_idx = _UMETRACK_IDX.to(device)

    def _make_optimizer():
        return torch.optim.LBFGS(
            opt_params, lr=0.1, max_iter=50,
            line_search_fn="strong_wolfe", history_size=30,
            tolerance_grad=1e-9, tolerance_change=1e-11,
        )

    optimizer = _make_optimizer()

    def closure():
        optimizer.zero_grad()
        angles_20 = angle_min + angle_range * torch.sigmoid(raw_angles)
        a22 = torch.cat([angles_20, torch.zeros(B, 2, device=device)], dim=1)

        pred_mesh, pred_lm = None, None
        if use_ch and use_lm:
            pred_mesh, pred_lm = umetrack_fk_batch(hand_model, a22, device, return_mesh=True, return_landmarks=True)
        elif use_ch:
            pred_mesh = umetrack_fk_batch(hand_model, a22, device, return_mesh=True)
        elif use_lm:
            pred_lm = umetrack_fk_batch(hand_model, a22, device, return_mesh=False, return_landmarks=True)

        loss = torch.tensor(0.0, device=device)
        if use_ch:
            loss = loss + chamfer_weight * chamfer_sym_sampled(
                pred_mesh, _transform(mano_v), ut_vidx, mano_vidx)
        if use_lm:
            loss = loss + landmark_weight * landmark_l2_batch(
                pred_lm[:, utrack_idx, :], _transform(mano_j)[:, mano_idx, :])
        loss.backward()
        return loss

    prev_loss = float("inf")
    best_loss = float("inf")
    stale_steps = 0
    patience = 30
    for _step in range(max_iter):
        loss = optimizer.step(closure)
        loss_v = loss.item()
        # Early stop if loss flatlines.
        if loss_v < best_loss - 0.0001:
            best_loss = loss_v
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= patience:
            break
        if _step > 0 and _step % 30 == 0 and abs(loss_v - prev_loss) < 0.01:
            optimizer = _make_optimizer()
            stale_steps = 0  # reset on restart
        prev_loss = loss_v

    with torch.no_grad():
        fa = angle_min + angle_range * torch.sigmoid(raw_angles)
        a22 = torch.cat([fa, torch.zeros(B, 2, device=device)], dim=1)
        pred_mesh, pred_lm = None, None
        if use_ch and use_lm:
            pred_mesh, pred_lm = umetrack_fk_batch(hand_model, a22, device, return_mesh=True, return_landmarks=True)
        elif use_ch:
            pred_mesh = umetrack_fk_batch(hand_model, a22, device, return_mesh=True)
        elif use_lm:
            pred_lm = umetrack_fk_batch(hand_model, a22, device, return_mesh=False, return_landmarks=True)

        fl = torch.tensor(0.0, device=device)
        if use_ch:
            fl = fl + chamfer_weight * chamfer_sym_sampled(
                pred_mesh, _transform(mano_v), ut_vidx, mano_vidx)
        if use_lm:
            fl = fl + landmark_weight * landmark_l2_batch(
                pred_lm[:, utrack_idx, :], _transform(mano_j)[:, mano_idx, :])
    return fa.cpu().numpy().astype(np.float32), fl.item()


# ═══════════════════════════════════════════════════════════════════════════════
# GPU worker
# ═══════════════════════════════════════════════════════════════════════════════

def _worker(gpu_id, memmap_root, hand, start_frame, end_frame,
            max_iter, batch_size, subsample_step,
            loss_type, chamfer_weight, landmark_weight, chamfer_verts,
            opt_global_orient, opt_trans):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    memmap_root = Path(memmap_root)

    # Load data memmaps.
    with open(memmap_root / "manifest.json") as f:
        manifest = json.load(f)
    pose_info = manifest["fields"][f"generated_mano_{hand}_pose"]
    pose_mm = np.memmap(memmap_root / pose_info["filename"],
                        dtype=np.dtype(pose_info["dtype"]), mode="r",
                        shape=tuple(pose_info["shape"]))
    out_mm = np.memmap(memmap_root / f"generated_joint_angles_{hand}.dat",
                       dtype=np.float32, mode="r+",
                       shape=(manifest["total_rows"], 20))

    # MANO setup.
    from egoemg.kinematics import apply_to_hand_model, load_default_hand_model

    mano_layer = ManoLayer(
        rot_mode="axisang", side="right",
        mano_assets_root=str(MANO_ASSETS_ROOT),
        use_pca=False, flat_hand_mean=True,
    ).to(device)

    hand_model = load_default_hand_model()
    hand_model = apply_to_hand_model(hand_model, lambda t: t.float().to(device))

    # Joint limits.
    json_path = (Path(__file__).resolve().parent.parent.parent
                 / "emg2pose" / "UmeTrack" / "dataset" / "generic_hand_model.json")
    with open(json_path) as f:
        limits = np.array(json.load(f)["joint_limits"][:20], dtype=np.float32)
    angle_min = torch.from_numpy(limits[:, 0]).to(device)
    angle_max = torch.from_numpy(limits[:, 1]).to(device)

    flip_t = torch.from_numpy(FLIP_MATRIX).float().to(device)
    mano_vidx = _uniform_idx(778, chamfer_verts).to(device)
    ut_vidx = _uniform_idx(788, chamfer_verts).to(device)

    # Load valid mask to skip WiLoR detection failures (NaN poses).
    valid_path = memmap_root / "valid.npy"
    valid_mask = np.load(valid_path).astype(bool) if valid_path.exists() else None
    if valid_mask is not None:
        n_bad = (~valid_mask[start_frame:end_frame]).sum()
        if n_bad:
            print(f"[GPU {gpu_id}] {hand}: {n_bad} invalid frames in [{start_frame}, {end_frame}) will be skipped")

    # DataLoader.
    dataset = AnchorPoseDataset(pose_mm, start_frame, end_frame, subsample_step, valid_mask)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_poses, pin_memory=True, drop_last=False,
    )

    n_anchors = len(dataset)
    n_processed = 0
    total_loss = 0.0
    t_start = time.time()
    report_interval = max(1, n_anchors // max(batch_size, 1) // 10)

    for batch_idx, (frame_indices, pose_batch) in enumerate(loader):
        pose_batch = pose_batch.to(device, non_blocking=True)
        pose_batch[:, :3] = 0.0  # zero global_orient → canonical MANO

        # Safety: skip frames with NaN poses (shouldn't happen if valid_mask works).
        nan_mask = torch.isnan(pose_batch).any(dim=1)
        if nan_mask.any():
            n_nan = int(nan_mask.sum().item())
            good_mask = ~nan_mask
            frame_indices = frame_indices[good_mask]
            pose_batch = pose_batch[good_mask]
            if pose_batch.shape[0] == 0:
                continue
            print(f"  [GPU {gpu_id}] WARNING: {n_nan} NaN poses detected at batch {batch_idx}, skipping")

        B_actual = pose_batch.shape[0]

        angles_batch, loss_val = optimize_batch(
            hand_model=hand_model,
            mano_layer=mano_layer,
            pose_batch=pose_batch,
            flip_t=flip_t,
            angle_min=angle_min,
            angle_max=angle_max,
            max_iter=max_iter,
            opt_global_orient=opt_global_orient,
            opt_trans=opt_trans,
            loss_type=loss_type,
            chamfer_weight=chamfer_weight,
            landmark_weight=landmark_weight,
            mano_vidx=mano_vidx,
            ut_vidx=ut_vidx,
            device=device,
        )

        for i, fi in enumerate(frame_indices):
            out_mm[int(fi.item())] = angles_batch[i]

        n_processed += B_actual
        total_loss += loss_val * B_actual

        if batch_idx % report_interval == 0 or n_processed >= n_anchors:
            elapsed = time.time() - t_start
            fps = n_processed / max(elapsed, 0.01)
            print(f"[GPU {gpu_id}] {hand} {n_processed}/{n_anchors} anchors "
                  f"({100*n_processed/n_anchors:.0f}%) {fps:.1f} fps, "
                  f"loss={total_loss/n_processed:.4f}, elapsed={elapsed:.1f}s")

    out_mm.flush()
    elapsed = time.time() - t_start
    print(f"[GPU {gpu_id}] {hand} DONE: [{start_frame}, {end_frame}) "
          f"({n_anchors} anchors) in {elapsed:.1f}s, "
          f"mean loss={total_loss/n_processed:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Interpolation
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_anchors(out_mm, total_rows, step):
    anchor_idx = np.arange(0, total_rows, step, dtype=np.int64)
    if anchor_idx[-1] < total_rows - 1:
        anchor_idx = np.append(anchor_idx, total_rows - 1)
    anchor_angles = out_mm[anchor_idx].copy()
    all_idx = np.arange(total_rows, dtype=np.float64)
    interp = np.empty((total_rows, 20), dtype=np.float32)
    for d in range(20):
        interp[:, d] = np.interp(all_idx, anchor_idx.astype(np.float64), anchor_angles[:, d])
    out_mm[:] = interp
    out_mm.flush()
    print(f"Interpolated {len(anchor_idx)} anchors → {total_rows} frames (step={step})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _process_hand(hand, gpu_ids, memmap_root, total_rows,
                  max_iter, batch_size, subsample_step,
                  loss_type, chamfer_weight, landmark_weight, chamfer_verts,
                  opt_global_orient, opt_trans):
    num_gpus = len(gpu_ids)
    out_path = memmap_root / f"generated_joint_angles_{hand}.dat"

    if not out_path.exists():
        mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(total_rows, 20))
        mm[:] = 0.0; mm.flush(); del mm
        print(f"[{hand}] Created output memmap: {out_path}")
    else:
        print(f"[{hand}] Output memmap exists (will overwrite assigned range)")

    frames_per_gpu = total_rows // num_gpus
    mp_ctx = mp.get_context("spawn")
    processes = []

    for i, gpu_id in enumerate(gpu_ids):
        start = i * frames_per_gpu
        end = start + frames_per_gpu if i < num_gpus - 1 else total_rows
        print(f"  GPU {gpu_id}: [{start}, {end}) ({end - start} frames)")
        p = mp_ctx.Process(target=_worker, kwargs=dict(
            gpu_id=gpu_id, memmap_root=str(memmap_root), hand=hand,
            start_frame=start, end_frame=end,
            max_iter=max_iter, batch_size=batch_size, subsample_step=subsample_step,
            loss_type=loss_type, chamfer_weight=chamfer_weight,
            landmark_weight=landmark_weight, chamfer_verts=chamfer_verts,
            opt_global_orient=opt_global_orient, opt_trans=opt_trans,
        ))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        sys.exit(1)

    if subsample_step > 1:
        out_mm = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(total_rows, 20))
        interpolate_anchors(out_mm, total_rows, subsample_step)

    with open(memmap_root / "manifest.json") as f:
        manifest = json.load(f)
    manifest["fields"][f"generated_joint_angles_{hand}"] = {
        "filename": f"generated_joint_angles_{hand}.dat",
        "dtype": "float32", "shape": [total_rows, 20],
    }
    with open(memmap_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{hand}] Updated manifest")


def main():
    parser = argparse.ArgumentParser(description="Batch IK mesh optimization")
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated GPU ids (e.g. 0,1,2,3)")
    parser.add_argument("--hand", default=None, choices=["left", "right"],
                        help="Process single hand only (default: both)")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10096)
    parser.add_argument("--subsample-step", type=int, default=1,
                        help="Solve IK on every Nth frame, interpolate the rest.")
    parser.add_argument("--loss-type", default="landmark",
                        choices=["chamfer", "landmark", "both"])
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--landmark-weight", type=float, default=1.0)
    parser.add_argument("--chamfer-verts", type=int, default=100)
    parser.add_argument("--no-opt-global-orient", dest="opt_global_orient",
                        action="store_false", default=True)
    parser.add_argument("--no-opt-trans", dest="opt_trans",
                        action="store_false", default=True)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit total frames (for dry-run testing).")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    print(f"GPUs: {gpu_ids}  |  max_iter={args.max_iter}  |  batch_size={args.batch_size}")
    print(f"subsample_step={args.subsample_step}  |  loss={args.loss_type}  |  "
          f"chamfer_verts={args.chamfer_verts}")
    print(f"opt_global_orient={args.opt_global_orient}  opt_trans={args.opt_trans}")
    print(f"Alignment: scale={ALIGN_SCALE:.4f}  trans={ALIGN_TRANS}")

    with open(args.memmap_root / "manifest.json") as f:
        manifest = json.load(f)
    total_rows = int(manifest["total_rows"])
    if args.max_frames is not None:
        total_rows = min(total_rows, args.max_frames)
    print(f"Total frames: {total_rows}")

    # Write joint angle semantics once.
    manifest["generated_joint_angles_semantics"] = [
        "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe",
        "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
        "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
        "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
        "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
    ]
    with open(args.memmap_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    hands = [args.hand] if args.hand else ["right", "left"]
    print(f"Processing: {hands}")

    for hand in hands:
        print(f"\n{'='*60}\nHand: {hand}\n{'='*60}")
        _process_hand(
            hand=hand, gpu_ids=gpu_ids, memmap_root=args.memmap_root,
            total_rows=total_rows,
            max_iter=args.max_iter, batch_size=args.batch_size,
            subsample_step=args.subsample_step,
            loss_type=args.loss_type, chamfer_weight=args.chamfer_weight,
            landmark_weight=args.landmark_weight, chamfer_verts=args.chamfer_verts,
            opt_global_orient=args.opt_global_orient, opt_trans=args.opt_trans,
        )

    print("\nAll done!")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    main()
