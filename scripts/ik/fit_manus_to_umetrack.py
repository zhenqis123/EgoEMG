#!/usr/bin/env python3
"""Fit UmeTrack 20D angles directly to Manus glove keypoints via L-BFGS.

Pipeline:
  1. Load Manus JSONL keypoints (25 nodes, meters, fingers +Z)
  2. Convert m→mm, map to UmeTrack 20 landmarks (L0-L19), rotate frame (→ fingers +X)
  3. Per-anchor-frame L-BFGS: optimize 20D finger angles + 3D wrist rotation to
     minimize L2(FK(θ)[:20]*scale, Manus_targets)
  4. Interpolate non-anchor frames, align with EMG timestamps
  5. Export (T, 20) float32 for training

Usage:
  python scripts/ik/fit_manus_to_umetrack.py \
    --session data/data/data_20260525_180032 \
    --output data/manus_fit

  python scripts/ik/fit_manus_to_umetrack.py \
    --session data/data/data_20260525_171835 \
    --output data/manus_fit --step 5
"""

from __future__ import annotations

import argparse
import json
import time as time_mod
from pathlib import Path

import numpy as np
import torch

from umetrack_fk_utils import (
    LM_WEIGHTS,
    angles_20d_to_22d,
    estimate_scale,
    extract_manus_targets,
    fk_landmarks,
    get_joint_limits,
    hand_model_to,
    load_model,
    make_wrist_transform,
)


def load_manus_frames(jsonl_path: Path) -> list[dict]:
    frames = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def optimize_frame_angles(
    targets: torch.Tensor,
    hand_model,
    init_angles: torch.Tensor,
    init_wrist_aa: torch.Tensor,
    init_scale: float,
    init_trans: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lm_weights: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.1,
    history_size: int = 20,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, float]:
    """Optimize 20D angles + 3D wrist rotation + scale + 3D translation.

    Returns (angles_20d, wrist_aa_3d, scale, translation_3d, final_loss).
    """
    d = torch.device(device)
    params = torch.cat([
        init_angles.detach().clone().to(d),
        init_wrist_aa.detach().clone().to(d),
        torch.tensor([init_scale], device=d),
        init_trans.detach().clone().to(d),
    ])
    params.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [params],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        with torch.no_grad():
            params[:20].clamp_(lower, upper)
            params[23].clamp_(0.3, 2.0)
            params[24:27].clamp_(-50, 50)
        full = angles_20d_to_22d(params[:20])
        wrist_tf = make_wrist_transform(params[20:23], params[24:27])
        lm = fk_landmarks(full, hand_model, wrist_transform=wrist_tf)[:20] * params[23]
        diff = lm - targets
        loss = (diff.pow(2).sum(dim=-1) * lm_weights).mean()
        reg = 1e-5 * params[:20].pow(2).sum()
        total = loss + reg
        total.backward()
        return total

    optimizer.step(closure)
    with torch.no_grad():
        params[:20].clamp_(lower, upper)
        params[23].clamp_(0.3, 2.0)
        params[24:27].clamp_(-50, 50)
    final_loss = closure().item()
    return (params[:20].detach(), params[20:23].detach(), params[23].item(),
            params[24:27].detach(), final_loss)


def process_session(
    session_dir: Path,
    output_dir: Path,
    step: int = 10,
    max_frames: int | None = None,
    device: str = "cpu",
) -> dict:
    session_name = session_dir.name
    session_out = output_dir / session_name
    session_out.mkdir(parents=True, exist_ok=True)

    right_jsonl = session_dir / "manus_right.jsonl"
    if not right_jsonl.exists() or right_jsonl.stat().st_size == 0:
        print(f"  No right hand data in {session_name}, skipping")
        return {}
    frames = load_manus_frames(right_jsonl)
    if max_frames:
        frames = frames[:max_frames]
    print(f"  Loaded {len(frames)} Manus right frames")

    d = torch.device(device)
    hand_model = load_model()
    hand_model = hand_model_to(hand_model, d)
    lower, upper = get_joint_limits(hand_model)
    lower, upper = lower.to(d), upper.to(d)

    # Estimate global scale from first frame (in UmeTrack frame after rotation)
    first_kp = np.array(frames[0]["keypoints"], dtype=np.float32) * 1000  # m→mm
    first_targets = extract_manus_targets(first_kp)
    scale = estimate_scale(first_kp, hand_model)
    print(f"  Estimated hand scale: {scale:.4f}")

    # Quick check: loss at rest pose
    rest_lm = fk_landmarks(torch.zeros(22, device=d), hand_model)[:20] * scale
    rest_loss = ((rest_lm - first_targets.to(d)).pow(2).sum(dim=-1) * LM_WEIGHTS.to(d)).mean()
    print(f"  Rest-pose loss (frame 0): {rest_loss.item():.2f}")

    # Anchor frames
    n_frames = len(frames)
    anchor_indices = list(range(0, n_frames, step))
    if anchor_indices[-1] != n_frames - 1:
        anchor_indices.append(n_frames - 1)
    anchor_indices = sorted(set(anchor_indices))
    anchor_set = set(anchor_indices)
    print(f"  Anchor frames: {len(anchor_indices)} / {n_frames} (step={step})")

    # Initialize
    prev_angles = torch.zeros(20, device=d)
    prev_wrist_aa = torch.full((3,), 1e-4, device=d)
    prev_scale = scale
    prev_trans = torch.zeros(3, device=d)
    all_angles = np.zeros((n_frames, 20), dtype=np.float32)
    all_wrist_aa = np.zeros((n_frames, 3), dtype=np.float32)
    all_scales = np.full(n_frames, np.nan, dtype=np.float32)
    all_trans = np.zeros((n_frames, 3), dtype=np.float32)
    all_losses = np.full(n_frames, np.nan, dtype=np.float32)
    timestamps = np.array([f["local_ts"] for f in frames], dtype=np.float64)

    t_start = time_mod.time()
    for i in range(n_frames):
        if i in anchor_set:
            kp = np.array(frames[i]["keypoints"], dtype=np.float32) * 1000
            targets = extract_manus_targets(kp)

            angles, wrist_aa, opt_scale, opt_trans, loss = optimize_frame_angles(
                targets.to(d), hand_model, prev_angles.clone(), prev_wrist_aa.clone(),
                prev_scale, prev_trans.clone(), lower, upper, LM_WEIGHTS.to(d),
                max_iter=100, device=device,
            )
            all_angles[i] = angles.cpu().numpy()
            all_wrist_aa[i] = wrist_aa.cpu().numpy()
            all_scales[i] = opt_scale
            all_trans[i] = opt_trans.cpu().numpy()
            all_losses[i] = loss
            prev_angles = angles.clone().to(d)
            prev_wrist_aa = wrist_aa.clone().to(d)
            prev_scale = opt_scale
            prev_trans = opt_trans.clone().to(d)

        if (i + 1) % 500 == 0:
            elapsed = time_mod.time() - t_start
            fps = (i + 1) / elapsed
            valid = all_losses[~np.isnan(all_losses)]
            avg = np.mean(valid[-100:]) if len(valid) > 0 else float("nan")
            print(f"    Frame {i+1}/{n_frames} ({fps:.1f} fps), recent loss={avg:.4f}")

    # Fill non-anchor frames via linear interpolation between adjacent anchors
    for idx in range(len(anchor_indices) - 1):
        a, b = anchor_indices[idx], anchor_indices[idx + 1]
        for frame in range(a + 1, b):
            frac = (frame - a) / (b - a)
            all_angles[frame] = (1 - frac) * all_angles[a] + frac * all_angles[b]
            all_wrist_aa[frame] = (1 - frac) * all_wrist_aa[a] + frac * all_wrist_aa[b]
            all_scales[frame] = (1 - frac) * all_scales[a] + frac * all_scales[b]
            all_trans[frame] = (1 - frac) * all_trans[a] + frac * all_trans[b]

    elapsed = time_mod.time() - t_start
    anchor_losses = all_losses[~np.isnan(all_losses)]
    print(f"  Done in {elapsed:.1f}s ({n_frames/elapsed:.1f} fps total)")
    print(f"  Anchor losses: mean={np.mean(anchor_losses):.4f}, "
          f"median={np.median(anchor_losses):.4f}, max={np.max(anchor_losses):.4f}")

    # Save
    angles_path = session_out / "joint_angles_right.npy"
    wrist_aa_path = session_out / "wrist_rotation_right.npy"
    scales_path = session_out / "scales_right.npy"
    trans_path = session_out / "wrist_translation_right.npy"
    ts_path = session_out / "manus_timestamps.npy"
    np.save(angles_path, all_angles)
    np.save(wrist_aa_path, all_wrist_aa)
    np.save(scales_path, all_scales)
    np.save(trans_path, all_trans)
    np.save(ts_path, timestamps)

    anchor_scales = all_scales[~np.isnan(all_scales)]
    summary = {
        "session": session_name,
        "n_frames": n_frames,
        "n_anchors": len(anchor_indices),
        "step": step,
        "initial_scale": scale,
        "anchor_scale_mean": float(np.mean(anchor_scales)),
        "anchor_scale_std": float(np.std(anchor_scales)),
        "rest_pose_loss": rest_loss.item(),
        "anchor_loss_mean": float(np.mean(anchor_losses)),
        "anchor_loss_median": float(np.median(anchor_losses)),
        "anchor_loss_max": float(np.max(anchor_losses)),
        "elapsed_s": elapsed,
        "angles_file": str(angles_path),
        "wrist_rotation_file": str(wrist_aa_path),
        "scales_file": str(scales_path),
        "timestamps_file": str(ts_path),
    }
    with open(session_out / "fit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def _align_array(data: np.ndarray, manus_ts: np.ndarray, emg_ts: np.ndarray) -> np.ndarray:
    """Align data array with EMG timestamps via linear interpolation.

    Uses np.interp to linearly interpolate from Manus (120 Hz) to EMG (2000 Hz)
    timestamps, matching EgoEMG's label interpolation strategy.
    """
    if data.ndim == 1:
        return np.interp(emg_ts, manus_ts, data).astype(np.float32)
    result = np.zeros((len(emg_ts), data.shape[1]), dtype=np.float32)
    for d in range(data.shape[1]):
        result[:, d] = np.interp(emg_ts, manus_ts, data[:, d])
    return result


def align_with_emg(session_dir: Path, fit_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align fitted angles, wrist rotations, scales, and translations with EMG timestamps."""
    import csv

    angles = np.load(fit_dir / "joint_angles_right.npy")
    wrist_aa = np.load(fit_dir / "wrist_rotation_right.npy")
    scales = np.load(fit_dir / "scales_right.npy")
    trans = np.load(fit_dir / "wrist_translation_right.npy")
    manus_ts = np.load(fit_dir / "manus_timestamps.npy")

    emg_ts = []
    with open(session_dir / "emg.csv") as f:
        for row in csv.DictReader(f):
            emg_ts.append(float(row["timestamp"]))
    emg_ts = np.array(emg_ts, dtype=np.float64)

    aligned_angles = _align_array(angles, manus_ts, emg_ts)
    aligned_wrist = _align_array(wrist_aa, manus_ts, emg_ts)
    aligned_scales = _align_array(scales, manus_ts, emg_ts)
    aligned_trans = _align_array(trans, manus_ts, emg_ts)

    max_dt = np.max(np.abs(manus_ts[np.clip(np.searchsorted(manus_ts, emg_ts), 0, len(manus_ts) - 1)] - emg_ts)) * 1000
    print(f"  EMG: {len(emg_ts)} samples, Manus: {len(manus_ts)} frames, max dt={max_dt:.2f} ms")
    return aligned_angles, aligned_wrist, aligned_scales, aligned_trans


def main():
    parser = argparse.ArgumentParser(description="Fit UmeTrack angles to Manus keypoints")
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", default="data/manus_fit")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    session_dir = Path(args.session).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {session_dir}  (device={args.device})")
    summary = process_session(session_dir, output_dir, step=args.step,
                              max_frames=args.max_frames, device=args.device)

    if not summary:
        print("No data processed")
        return

    print("\nAligning with EMG...")
    session_name = session_dir.name
    fit_dir = output_dir / session_name
    aligned_angles, aligned_wrist, aligned_scales, aligned_trans = align_with_emg(session_dir, fit_dir)

    aligned_path = fit_dir / "joint_angles_right_emg_aligned.npy"
    wrist_aligned_path = fit_dir / "wrist_rotation_right_emg_aligned.npy"
    scales_aligned_path = fit_dir / "scales_right_emg_aligned.npy"
    trans_aligned_path = fit_dir / "wrist_translation_right_emg_aligned.npy"
    np.save(aligned_path, aligned_angles)
    aligned_angles.tofile(fit_dir / "joint_angles_right_emg_aligned.dat")
    np.save(wrist_aligned_path, aligned_wrist)
    np.save(scales_aligned_path, aligned_scales)
    np.save(trans_aligned_path, aligned_trans)
    print(f"  Saved: {aligned_path} ({aligned_angles.shape})")
    print(f"  Saved: {wrist_aligned_path} ({aligned_wrist.shape})")
    print(f"  Saved: {scales_aligned_path} ({aligned_scales.shape})")
    print(f"  Saved: {trans_aligned_path} ({aligned_trans.shape})")

    print(f"\nDone. Output in {fit_dir}/")


if __name__ == "__main__":
    main()
