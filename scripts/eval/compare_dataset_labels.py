#!/usr/bin/env python3
"""
Compare label quality between EgoEmg and EMG2Pose datasets.

Three analysis dimensions:
1. Pose space diversity — per-joint range, PCA, wrist DoF (random sampling OK)
2. IK failure rate — fraction of invalid frames (random sampling OK)
3. Label smoothness — contiguous-block outlier analysis, ~120Hz decimated jerk

Usage:
    python scripts/eval/compare_dataset_labels.py \
        --emg2pose-dir /path/to/emg2pose_memmap \
        --egoemg-dir /path/to/EgoEMG_v2_memmap \
        --max-frames 200000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

JOINT_NAMES = [
    "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
    "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
    "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
    "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
    "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
]

FINGER_GROUPS = {
    "Thumb": [0, 1, 2, 3],
    "Index": [4, 5, 6, 7],
    "Middle": [8, 9, 10, 11],
    "Ring": [12, 13, 14, 15],
    "Pinky": [16, 17, 18, 19],
}


# ─── data loading ────────────────────────────────────────────────

def _open_memmap(mm_dir, info):
    return np.memmap(
        mm_dir / info["filename"],
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def load_emg2pose_meta(memmap_dir):
    """Return memmap arrays + block metadata for EMG2Pose."""
    mm_dir = Path(memmap_dir)
    manifest = json.loads((mm_dir / "manifest.json").read_text())
    fields = manifest["fields"]
    meta = np.load(mm_dir / "metadata.npz", allow_pickle=True)

    ja = _open_memmap(mm_dir, fields["joint_angles"])  # (N, 20)
    ts = _open_memmap(mm_dir, fields["time"])           # (N,)
    vm = _open_memmap(mm_dir, fields["valid_mask"])     # (N,)

    blocks_start = meta["blocks_start"]
    blocks_end = meta["blocks_end"]

    return ja, vm, ts, blocks_start, blocks_end


def load_egoemg_meta(memmap_dir):
    """Return memmap arrays + episode metadata for EgoEmg."""
    mm_dir = Path(memmap_dir)
    manifest = json.loads((mm_dir / "manifest.json").read_text())
    fields = manifest["fields"]

    ja_l = _open_memmap(mm_dir, fields["generated_joint_angles_left"])
    ja_r = _open_memmap(mm_dir, fields["generated_joint_angles_right"])
    vm_full = _open_memmap(mm_dir, fields["generated_label_valid"])  # (N, 2)
    ts = _open_memmap(mm_dir, fields["timestamp"])
    ep_idx = _open_memmap(mm_dir, fields["episode_index"])

    return ja_l, ja_r, vm_full, ts, ep_idx


# ─── random-frame metrics (validity, range, PCA) ─────────────────

def sample_random_frames(memmap_arrays, memmap_masks, max_frames, seed):
    """Sample random indices from memmaps and return those frames + global indices.

    Args:
        memmap_arrays: list of np.memmap (N_i, 20) — joint angle memmaps
        memmap_masks: list of np.memmap (N_i,) — valid mask memmaps (1:1 with arrays)
    Returns:
        ja_sample: (sampled, 20), vm_sample: (sampled,), global_idx: (sampled,)
    """
    total = sum(len(mm) for mm in memmap_arrays)
    rng = np.random.RandomState(seed)
    if max_frames and total > max_frames:
        global_idx = np.sort(rng.choice(total, size=max_frames, replace=False))
    else:
        global_idx = np.arange(total)

    offsets = np.cumsum([0] + [len(mm) for mm in memmap_arrays[:-1]])
    ja_chunks, vm_chunks = [], []
    for mm_ja, mm_vm, offset in zip(memmap_arrays, memmap_masks, offsets):
        local_idx = global_idx[(global_idx >= offset) & (global_idx < offset + len(mm_ja))]
        if len(local_idx) == 0:
            continue
        local_idx = np.sort((local_idx - offset).astype(np.int64))
        ja_chunks.append(np.array(mm_ja[local_idx]))
        vm_chunks.append(np.array(mm_vm[local_idx]))
    return np.concatenate(ja_chunks), np.concatenate(vm_chunks), global_idx


def compute_validity(angles, mask):
    n_total = len(mask)
    n_valid = int(mask.sum())
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_total - n_valid,
        "invalid_rate": 1.0 - n_valid / n_total,
    }


def compute_joint_stats(angles, mask):
    valid = angles[mask]
    valid = valid[np.isfinite(valid).all(axis=1)]
    return {
        "joint_min": valid.min(axis=0).tolist(),
        "joint_max": valid.max(axis=0).tolist(),
        "joint_range": (valid.max(axis=0) - valid.min(axis=0)).tolist(),
        "joint_std": valid.std(axis=0).tolist(),
        "total_range": float((valid.max(axis=0) - valid.min(axis=0)).sum()),
    }


def compute_pca_stats(angles, mask, max_samples=50000):
    from sklearn.decomposition import PCA

    valid = angles[mask]
    valid = valid[np.isfinite(valid).all(axis=1)]
    if len(valid) > max_samples:
        valid = valid[np.random.RandomState(42).choice(len(valid), size=max_samples, replace=False)]

    pca = PCA().fit(valid)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumsum, 0.95) + 1)
    return {
        "n_components_95": n95,
        "n_components_total": len(pca.explained_variance_ratio_),
        "explained_variance_top5": pca.explained_variance_ratio_[:5].tolist(),
        "total_variance": float(np.sum(pca.explained_variance_)),
    }


# ─── gesture/stage diversity metrics ──────────────────────────────

def load_emg2pose_stage_info(memmap_dir):
    """Load stage metadata from EMG2Pose metadata.npz."""
    meta = np.load(Path(memmap_dir) / "metadata.npz", allow_pickle=True)
    return {
        "stage_ids": meta["stages_stage_id"],
        "stage_names": [s.decode() if isinstance(s, bytes) else str(s)
                        for s in meta["stages_stage"]],
        "session_stage_id": meta["session_stage_id"],
        "session_start": meta["session_start_idx"],
        "session_end": meta["session_end_idx"],
    }


def map_frames_to_stages(global_idx, session_start, session_end, session_stage_id):
    """Map global frame indices to stage IDs via session boundaries.

    Uses binary search to find which session each frame belongs to.
    Returns stage_id for each frame (-1 if not found).
    """
    # session_start is sorted; find session index for each frame
    sess_idx = np.searchsorted(session_end, global_idx, side="right")
    # Clamp to valid range
    sess_idx = np.clip(sess_idx, 0, len(session_start) - 1)
    # Verify each frame is within its assigned session
    in_range = (global_idx >= session_start[sess_idx]) & (global_idx < session_end[sess_idx])
    stage_id = np.full(len(global_idx), -1, dtype=np.int32)
    stage_id[in_range] = session_stage_id[sess_idx[in_range]]
    return stage_id


def compute_inter_gesture_separation(ja, vm, class_labels, class_valid_mask=None):
    """Metric 2: Inter-gesture/stage pose separation.

    Computes per-class mean poses and pairwise Euclidean distances.

    Args:
        ja: (N, 20) joint angles
        vm: (N,) valid mask (label is valid)
        class_labels: (N,) int class IDs
        class_valid_mask: (N,) optional bool — only use frames where True
          (e.g., label_gesture_active for EgoEmg)

    Returns:
        dict with per-class means, pairwise distance stats
    """
    # Combine validity: label valid AND (optionally) gesture active
    valid = vm.copy()
    if class_valid_mask is not None:
        valid = valid & class_valid_mask

    # Also filter non-finite joint angles
    valid = valid & np.isfinite(ja).all(axis=1)

    # Get unique classes (excluding -1 / background)
    unique_classes = np.unique(class_labels[valid])
    unique_classes = unique_classes[unique_classes >= 0]

    # Compute per-class mean; skip classes with too few samples
    per_class_means = {}
    min_samples = 20
    for cls in unique_classes:
        cls_mask = valid & (class_labels == cls)
        cls_ja = ja[cls_mask]
        if len(cls_ja) < min_samples:
            continue
        per_class_means[int(cls)] = cls_ja.mean(axis=0)

    if len(per_class_means) < 2:
        return {"error": f"Only {len(per_class_means)} classes with ≥{min_samples} samples"}

    class_ids = list(per_class_means.keys())
    means = np.array([per_class_means[c] for c in class_ids])

    # Pairwise Euclidean distances
    distances = []
    for i in range(len(class_ids)):
        for j in range(i + 1, len(class_ids)):
            distances.append(float(np.linalg.norm(means[i] - means[j])))

    distances = np.array(distances)
    return {
        "n_classes": len(class_ids),
        "n_classes_total": len(unique_classes),
        "class_ids": class_ids,
        "mean_pairwise_distance": float(distances.mean()),
        "min_pairwise_distance": float(distances.min()),
        "max_pairwise_distance": float(distances.max()),
        "std_pairwise_distance": float(distances.std()),
        "median_pairwise_distance": float(np.median(distances)),
    }


def compute_intra_gesture_variance(ja, vm, class_labels, class_valid_mask=None):
    """Metric 3: Intra-gesture variance & variance-explained-by-gesture ratio.

    Computes:
      - Mean within-class variance (trace of covariance)
      - Total variance of all valid frames
      - Fraction of total variance explained by gesture identity
        (= 1 - within_var / total_var, i.e., how much variance is BETWEEN classes)

    Args:
        ja: (N, 20) joint angles
        vm: (N,) valid mask
        class_labels: (N,) int class IDs
        class_valid_mask: (N,) optional — additional mask (e.g., gesture_active)

    Returns:
        dict with variance stats
    """
    valid = vm.copy()
    if class_valid_mask is not None:
        valid = valid & class_valid_mask
    valid = valid & np.isfinite(ja).all(axis=1)

    unique_classes = np.unique(class_labels[valid])
    unique_classes = unique_classes[unique_classes >= 0]

    min_samples = 20
    within_vars = []
    class_frame_counts = []

    for cls in unique_classes:
        cls_mask = valid & (class_labels == cls)
        cls_ja = ja[cls_mask]
        if len(cls_ja) < min_samples:
            continue
        # Trace of covariance = sum of per-joint variances
        cls_var = np.var(cls_ja, axis=0).sum()
        within_vars.append(cls_var)
        class_frame_counts.append(len(cls_ja))

    if len(within_vars) < 2:
        return {"error": f"Only {len(within_vars)} classes with ≥{min_samples} samples"}

    within_vars = np.array(within_vars)
    class_frame_counts = np.array(class_frame_counts)

    # Weight within-class variance by class size
    mean_within_var = float(np.average(within_vars, weights=class_frame_counts))

    # Total variance across all valid frames used in class analysis
    all_class_mask = np.zeros(len(ja), dtype=bool)
    for cls in unique_classes:
        cls_mask = valid & (class_labels == cls)
        if cls_mask.sum() >= min_samples:
            all_class_mask = all_class_mask | cls_mask
    total_var = float(np.var(ja[all_class_mask], axis=0).sum())

    # Fraction of variance explained by class identity
    # (= between-class variance / total variance)
    var_explained = 1.0 - mean_within_var / total_var if total_var > 0 else 0.0

    return {
        "n_classes_used": len(within_vars),
        "n_classes_total": len(unique_classes),
        "mean_within_class_variance": mean_within_var,
        "total_variance": total_var,
        "variance_explained_by_class": var_explained,
        "class_frame_counts": class_frame_counts.tolist(),
    }


def compute_wrist_coverage(memmap_dir, max_frames=50000, seed=42):
    """Metric 5: Wrist articulation coverage (EgoEmg only).

    Computes 2D histogram and convex hull area of wrist F/E vs R/U angles.

    Returns:
        dict with coverage stats, suitable for paper figure
    """
    mm_dir = Path(memmap_dir)
    manifest = json.loads((mm_dir / "manifest.json").read_text())
    fields = manifest["fields"]

    # Load wrist pitch (F/E) and yaw (R/U) for both hands
    pitch_r = _open_memmap(mm_dir, fields["mocap_right_wrist_pitch"])
    yaw_r = _open_memmap(mm_dir, fields["mocap_right_wrist_yaw"])
    valid_r = _open_memmap(mm_dir, fields["mocap_right_wrist_angles_valid"])

    pitch_l = _open_memmap(mm_dir, fields["mocap_left_wrist_pitch"])
    yaw_l = _open_memmap(mm_dir, fields["mocap_left_wrist_yaw"])
    valid_l = _open_memmap(mm_dir, fields["mocap_left_wrist_angles_valid"])

    # Sample valid frames
    total = len(pitch_r)
    rng = np.random.RandomState(seed)
    if total > max_frames:
        idx = np.sort(rng.choice(total, size=max_frames, replace=False))
    else:
        idx = np.arange(total)

    # Collect valid wrist angles from sampled frames
    pitch_vals, yaw_vals = [], []
    for hand_valid, pitch_mm, yaw_mm in [(valid_r, pitch_r, yaw_r),
                                           (valid_l, pitch_l, yaw_l)]:
        hand_idx = idx[hand_valid[idx]]
        if len(hand_idx) == 0:
            continue
        # Raw wrist angles stored in degrees; convert to radians
        p = np.deg2rad(np.array(pitch_mm[hand_idx]))
        y = np.deg2rad(np.array(yaw_mm[hand_idx]))
        finite_mask = np.isfinite(p) & np.isfinite(y)
        pitch_vals.append(p[finite_mask])
        yaw_vals.append(y[finite_mask])

    if not pitch_vals:
        return {"error": "No valid wrist angle frames found"}

    all_pitch = np.concatenate(pitch_vals)
    all_yaw = np.concatenate(yaw_vals)

    # Compute stats from wrapped angles
    pitch_range = float(all_pitch.max() - all_pitch.min())
    yaw_range = float(all_yaw.max() - all_yaw.min())

    # Convex hull area (2D)
    try:
        from scipy.spatial import ConvexHull
        points = np.column_stack([all_pitch, all_yaw])
        if len(points) > 10000:
            points = points[rng.choice(len(points), size=10000, replace=False)]
        hull = ConvexHull(points)
        hull_area = float(hull.volume)  # In 2D, hull.volume = area
    except ImportError:
        hull_area = float("nan")

    return {
        "n_frames": len(all_pitch),
        "pitch_min_deg": float(np.rad2deg(all_pitch.min())),
        "pitch_max_deg": float(np.rad2deg(all_pitch.max())),
        "pitch_range_deg": float(np.rad2deg(pitch_range)),
        "yaw_min_deg": float(np.rad2deg(all_yaw.min())),
        "yaw_max_deg": float(np.rad2deg(all_yaw.max())),
        "yaw_range_deg": float(np.rad2deg(yaw_range)),
        "convex_hull_area_rad2": hull_area,
        "pitch_mean_deg": float(np.rad2deg(all_pitch.mean())),
        "pitch_std_deg": float(np.rad2deg(all_pitch.std())),
        "yaw_mean_deg": float(np.rad2deg(all_yaw.mean())),
        "yaw_std_deg": float(np.rad2deg(all_yaw.std())),
        "n_left_frames": len(pitch_vals[1]) if len(pitch_vals) > 1 else 0,
        "n_right_frames": len(pitch_vals[0]),
    }


# ─── contiguous-block smoothness ─────────────────────────────────

def sample_contiguous_blocks(ja, vm, ts, block_starts, block_ends, n_blocks, seed):
    """Sample n_blocks contiguous segments with high valid-frame density."""
    rng = np.random.RandomState(seed)
    # Filter blocks by valid rate and length
    candidates = []
    for s, e in zip(block_starts, block_ends):
        length = e - s
        if length < 200:
            continue
        valid_rate = vm[s:e].mean()
        if valid_rate < 0.5:
            continue
        candidates.append((int(s), int(e), valid_rate))

    if len(candidates) < n_blocks:
        n_blocks = len(candidates)

    chosen_idx = rng.choice(len(candidates), size=n_blocks, replace=False)
    blocks = []
    for i in chosen_idx:
        s, e, _ = candidates[i]
        blocks.append({
            "ja": np.array(ja[s:e]),
            "vm": np.array(vm[s:e]),
            "ts": np.array(ts[s:e]),
            "length": e - s,
        })
    return blocks


def sample_egoemg_contiguous_blocks(ja, vm, ts, ep_idx, n_blocks, seed):
    """Sample contiguous segments from EgoEmg using episode boundaries."""
    rng = np.random.RandomState(seed)

    # Find episode transitions
    unique_eps = np.unique(ep_idx)
    candidates = []
    for ep in unique_eps:
        ep_mask = ep_idx == ep
        ep_indices = np.where(ep_mask)[0]
        if len(ep_indices) < 200:
            continue
        # Take a random contiguous window from this episode
        max_start = len(ep_indices) - 500
        if max_start <= 0:
            continue
        for _ in range(3):  # try up to 3 windows per episode
            w_start = rng.randint(0, min(max_start, 2000))
            w_len = min(rng.randint(200, 2000), len(ep_indices) - w_start)
            s = ep_indices[w_start]
            e = ep_indices[w_start + w_len]
            valid_rate = vm[s:e].mean()
            if valid_rate > 0.5:
                candidates.append((int(s), int(e), valid_rate))
                break

    if len(candidates) < n_blocks:
        n_blocks = len(candidates)

    chosen_idx = rng.choice(len(candidates), size=n_blocks, replace=False)
    blocks = []
    for i in chosen_idx:
        s, e, _ = candidates[i]
        blocks.append({
            "ja": np.array(ja[s:e]),
            "vm": np.array(vm[s:e]),
            "ts": np.array(ts[s:e]),
            "length": e - s,
        })
    return blocks


def compute_block_smoothness(ja, vm, ts, decimate=16):
    """Compute smoothness within one contiguous block.

    Returns aggregate stats for outlier rates and decimated smoothness.
    """
    valid_idx = np.where(vm)[0]
    if len(valid_idx) < 10:
        return None

    # ── native-rate outlier analysis ──
    gaps = np.diff(valid_idx)
    consec = gaps == 1
    pair_starts = valid_idx[:-1][consec]
    pair_ends = valid_idx[1:][consec]

    if len(pair_starts) < 10:
        return None

    dtheta = np.abs(ja[pair_ends] - ja[pair_starts])
    max_step = dtheta.max(axis=1)

    # 1st diff
    step_mean = max_step.mean()
    step_std = max_step.std()
    step_outliers = int((max_step > step_mean + 3 * step_std).sum())
    step_total = len(max_step)

    # 2nd diff
    dtheta_2nd = np.abs(dtheta[1:] - dtheta[:-1])
    max_acc = dtheta_2nd.max(axis=1)
    acc_mean = max_acc.mean()
    acc_std = max_acc.std()
    acc_outliers = int((max_acc > acc_mean + 3 * acc_std).sum())
    acc_total = len(max_acc)

    # 3rd diff
    dtheta_3rd = np.abs(dtheta_2nd[1:] - dtheta_2nd[:-1])
    max_jerk = dtheta_3rd.max(axis=1)
    jerk_mean = max_jerk.mean()
    jerk_std = max_jerk.std()
    jerk_outliers = int((max_jerk > jerk_mean + 3 * jerk_std).sum())
    jerk_total = len(max_jerk)

    # Fraction of frames with zero change (stale labels)
    zero_change = int((max_step < 1e-7).sum())

    # ── decimated (~120Hz) time-normalized metrics ──
    dec_idx = valid_idx[::decimate]
    dec_results = {}
    if len(dec_idx) >= 4:
        dec_dt = ts[dec_idx[1:]] - ts[dec_idx[:-1]]
        valid_dt = (dec_dt > 1e-9) & (dec_dt < 10.0)
        if valid_dt.sum() >= 4:
            dec_vel = (ja[dec_idx[1:]] - ja[dec_idx[:-1]])[valid_dt] / dec_dt[valid_dt, None]
            dec_vel_abs = np.abs(dec_vel)
            dec_results["dec_n_pairs"] = int(valid_dt.sum())
            dec_results["dec_mean_vel"] = float(dec_vel_abs.mean())
            dec_results["dec_max_vel"] = float(dec_vel_abs.max())

            if len(dec_vel) >= 3:
                dec_dt_mid = 0.5 * (dec_dt[valid_dt][:-1] + dec_dt[valid_dt][1:])
                dec_acc = (dec_vel[1:] - dec_vel[:-1]) / dec_dt_mid[:, None]
                dec_acc_abs = np.abs(dec_acc)
                dec_results["dec_mean_acc"] = float(dec_acc_abs.mean())

            if len(dec_vel) >= 5:
                dec_dt_mid2 = 0.5 * (dec_dt_mid[:-1] + dec_dt_mid[1:])
                dec_jerk = (dec_acc[1:] - dec_acc[:-1]) / dec_dt_mid2[:, None]
                dec_results["dec_mean_jerk"] = float(np.abs(dec_jerk).mean())
            else:
                dec_results["dec_mean_jerk"] = float("nan")
        else:
            dec_results["dec_mean_vel"] = float("nan")
    else:
        dec_results["dec_mean_vel"] = float("nan")

    return {
        "block_length": len(ja),
        "n_valid_pairs": step_total,
        "step_outliers": step_outliers,
        "step_total": step_total,
        "acc_outliers": acc_outliers,
        "acc_total": acc_total,
        "jerk_outliers": jerk_outliers,
        "jerk_total": jerk_total,
        "zero_change_frames": zero_change,
        "mean_abs_step": float(max_step.mean()),
        "mean_abs_3rd_diff": float(max_jerk.mean()),
        **dec_results,
    }


def aggregate_smoothness(block_results):
    """Aggregate smoothness across multiple blocks."""
    valid = [r for r in block_results if r is not None]
    if not valid:
        return {"error": "No valid blocks"}

    total_step_outliers = sum(r["step_outliers"] for r in valid)
    total_step = sum(r["step_total"] for r in valid)
    total_acc_outliers = sum(r["acc_outliers"] for r in valid)
    total_acc = sum(r["acc_total"] for r in valid)
    total_jerk_outliers = sum(r["jerk_outliers"] for r in valid)
    total_jerk = sum(r["jerk_total"] for r in valid)
    total_zero = sum(r["zero_change_frames"] for r in valid)
    total_pairs = sum(r["n_valid_pairs"] for r in valid)

    mean_step = np.mean([r["mean_abs_step"] for r in valid])
    mean_3rd = np.mean([r["mean_abs_3rd_diff"] for r in valid])

    dec_vels = [r["dec_mean_vel"] for r in valid if not np.isnan(r.get("dec_mean_vel", float("nan")))]
    dec_accs = [r["dec_mean_acc"] for r in valid if not np.isnan(r.get("dec_mean_acc", float("nan")))]
    dec_jerks = [r["dec_mean_jerk"] for r in valid if not np.isnan(r.get("dec_mean_jerk", float("nan")))]

    return {
        "n_blocks": len(valid),
        "total_frames": sum(r["block_length"] for r in valid),
        "total_valid_pairs": total_pairs,
        "step_outlier_rate": total_step_outliers / total_step if total_step > 0 else 0,
        "acc_outlier_rate": total_acc_outliers / total_acc if total_acc > 0 else 0,
        "jerk_outlier_rate": total_jerk_outliers / total_jerk if total_jerk > 0 else 0,
        "zero_change_rate": total_zero / total_pairs if total_pairs > 0 else 0,
        "mean_abs_step": mean_step,
        "mean_abs_3rd_diff": mean_3rd,
        "dec_mean_vel": np.mean(dec_vels) if dec_vels else float("nan"),
        "dec_mean_acc": np.mean(dec_accs) if dec_accs else float("nan"),
        "dec_mean_jerk": np.mean(dec_jerks) if dec_jerks else float("nan"),
    }


# ─── display ─────────────────────────────────────────────────────

def print_table(headers, rows, aligns=None):
    if aligns is None:
        aligns = ["<"] * len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:{a}{w}}}" for a, w in zip(aligns, col_widths))
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + 2 * (len(headers) - 1)))
    for row in rows:
        print(fmt.format(*row))


# ─── main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare dataset label quality")
    parser.add_argument("--emg2pose-dir", required=True, type=str)
    parser.add_argument("--egoemg-dir", required=True, type=str)
    parser.add_argument("--max-frames", type=int, default=200000)
    parser.add_argument("--n-smooth-blocks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    # ── Load data ──
    print("Loading EMG2Pose...", file=sys.stderr)
    ja_emg, vm_emg, ts_emg, blk_start, blk_end = load_emg2pose_meta(args.emg2pose_dir)

    print("Loading EgoEmg...", file=sys.stderr)
    ja_ego_l, ja_ego_r, vm_ego_full, ts_ego, ep_idx = load_egoemg_meta(args.egoemg_dir)
    vm_ego_r = vm_ego_full[:, 1]  # memmap slice, not materialized

    # ── 1 & 2. Random-frame metrics (validity, range, PCA) ──
    print("Computing random-frame metrics...", file=sys.stderr)
    emg_ja_sample, emg_vm_sample, emg_global_idx = sample_random_frames(
        [ja_emg], [vm_emg], args.max_frames, args.seed
    )
    # For EgoEmg: pass left+right ja memmaps, and corresponding vm columns as separate memmaps
    ego_ja_sample, ego_vm_sample, ego_global_idx = sample_random_frames(
        [ja_ego_l, ja_ego_r],
        [vm_ego_full[:, 0], vm_ego_full[:, 1]],
        args.max_frames, args.seed,
    )

    validity_emg = compute_validity(emg_ja_sample, emg_vm_sample)
    validity_ego = compute_validity(ego_ja_sample, ego_vm_sample)
    joint_emg = compute_joint_stats(emg_ja_sample, emg_vm_sample)
    joint_ego = compute_joint_stats(ego_ja_sample, ego_vm_sample)
    pca_emg = compute_pca_stats(emg_ja_sample, emg_vm_sample)
    pca_ego = compute_pca_stats(ego_ja_sample, ego_vm_sample)

    # ── 4. Gesture diversity metrics ──
    print("Computing gesture diversity metrics...", file=sys.stderr)

    # EMG2Pose: map sampled frames to stage IDs
    stage_info = load_emg2pose_stage_info(args.emg2pose_dir)
    emg_stage_labels = map_frames_to_stages(
        emg_global_idx, stage_info["session_start"],
        stage_info["session_end"], stage_info["session_stage_id"],
    )
    sep_emg = compute_inter_gesture_separation(
        emg_ja_sample, emg_vm_sample, emg_stage_labels,
    )
    var_emg = compute_intra_gesture_variance(
        emg_ja_sample, emg_vm_sample, emg_stage_labels,
    )

    # EgoEmg: read gesture class + active labels for sampled frames
    ego_gc_mm = _open_memmap(
        Path(args.egoemg_dir),
        json.loads((Path(args.egoemg_dir) / "manifest.json").read_text())["fields"]["label_gesture_class"],
    )
    ego_ga_mm = _open_memmap(
        Path(args.egoemg_dir),
        json.loads((Path(args.egoemg_dir) / "manifest.json").read_text())["fields"]["label_gesture_active"],
    )
    # ego_global_idx covers both left (0..N-1) and right (N..2N-1);
    # gesture labels are indexed the same as left/right individually
    ego_gesture_class = np.array(ego_gc_mm[ego_global_idx % len(ego_gc_mm)])
    ego_gesture_active = np.array(ego_ga_mm[ego_global_idx % len(ego_ga_mm)])

    sep_ego = compute_inter_gesture_separation(
        ego_ja_sample, ego_vm_sample, ego_gesture_class, ego_gesture_active,
    )
    var_ego = compute_intra_gesture_variance(
        ego_ja_sample, ego_vm_sample, ego_gesture_class, ego_gesture_active,
    )

    # Wrist articulation (EgoEmg only)
    print("Computing wrist articulation coverage...", file=sys.stderr)
    wrist_ego = compute_wrist_coverage(args.egoemg_dir, seed=args.seed)

    # ── 3. Contiguous-block smoothness ──
    print("Computing contiguous-block smoothness...", file=sys.stderr)
    emg_blocks = sample_contiguous_blocks(
        ja_emg, vm_emg, ts_emg, blk_start, blk_end,
        args.n_smooth_blocks, args.seed,
    )
    ego_blocks = sample_egoemg_contiguous_blocks(
        ja_ego_r, vm_ego_r, ts_ego, ep_idx,
        args.n_smooth_blocks, args.seed,
    )

    emg_block_results = [compute_block_smoothness(b["ja"], b["vm"], b["ts"]) for b in emg_blocks]
    ego_block_results = [compute_block_smoothness(b["ja"], b["vm"], b["ts"]) for b in ego_blocks]

    smooth_emg = aggregate_smoothness(emg_block_results)
    smooth_ego = aggregate_smoothness(ego_block_results)

    if args.json:
        print(json.dumps({
            "validity": {"emg2pose": validity_emg, "egoemg": validity_ego},
            "joint_stats": {"emg2pose": joint_emg, "egoemg": joint_ego},
            "pca": {"emg2pose": pca_emg, "egoemg": pca_ego},
            "smoothness": {"emg2pose": smooth_emg, "egoemg": smooth_ego},
            "gesture_separation": {"emg2pose": sep_emg, "egoemg": sep_ego},
            "intra_gesture_variance": {"emg2pose": var_emg, "egoemg": var_ego},
            "wrist_coverage": wrist_ego,
        }, indent=2))
        return

    # === Output ===

    # 1. Validity
    print()
    print("=" * 64)
    print("1. LABEL VALIDITY — IK Failure Rate")
    print("=" * 64)
    print_table(
        ["Metric", "EMG2Pose", "EgoEmg"],
        [
            ("Frames sampled", f"{validity_emg['n_total']:,}", f"{validity_ego['n_total']:,}"),
            ("Valid frames", f"{validity_emg['n_valid']:,}", f"{validity_ego['n_valid']:,}"),
            ("Invalid (IK failed)", f"{validity_emg['n_invalid']:,}", f"{validity_ego['n_invalid']:,}"),
            ("Invalid rate", f"{validity_emg['invalid_rate']:.2%}", f"{validity_ego['invalid_rate']:.2%}"),
        ],
    )

    # 2. Pose Diversity
    print()
    print("=" * 64)
    print("2. POSE SPACE DIVERSITY")
    print("=" * 64)

    print("\n2a. Per-joint angular range (radians)")
    rows = []
    for i, name in enumerate(JOINT_NAMES):
        r_emg = joint_emg["joint_range"][i]
        r_ego = joint_ego["joint_range"][i]
        ratio = f"{r_ego / r_emg:.2f}x" if r_emg > 0.001 else "N/A"
        rows.append((name, f"{r_emg:.3f}", f"{r_ego:.3f}", ratio))
    print_table(["Joint", "EMG2Pose", "EgoEmg", "Ratio"], rows)

    print("\n2b. By finger group (radians)")
    fg_rows = []
    for fg_name, indices in FINGER_GROUPS.items():
        r_emg = sum(joint_emg["joint_range"][i] for i in indices)
        r_ego = sum(joint_ego["joint_range"][i] for i in indices)
        ratio = f"{r_ego / r_emg:.2f}x" if r_emg > 0.001 else "N/A"
        fg_rows.append((fg_name, f"{r_emg:.3f}", f"{r_ego:.3f}", ratio))
    r_total_emg = joint_emg["total_range"]
    r_total_ego = joint_ego["total_range"]
    fg_rows.append((
        "TOTAL (20 joints)", f"{r_total_emg:.2f}", f"{r_total_ego:.2f}",
        f"{r_total_ego / r_total_emg:.2f}x" if r_total_emg > 0 else "N/A",
    ))
    print_table(["Finger Group", "EMG2Pose", "EgoEmg", "Ratio"], fg_rows)

    print("\n2c. PCA (20 shared joints)")
    print_table(
        ["Metric", "EMG2Pose", "EgoEmg"],
        [
            ("PCs for 95% variance", str(pca_emg["n_components_95"]), str(pca_ego["n_components_95"])),
            (
                "Top-5 explained var",
                ", ".join(f"{v:.3f}" for v in pca_emg["explained_variance_top5"]),
                ", ".join(f"{v:.3f}" for v in pca_ego["explained_variance_top5"]),
            ),
            ("Total variance", f"{pca_emg['total_variance']:.4f}", f"{pca_ego['total_variance']:.4f}"),
        ],
    )

    print("\n2d. Wrist DoF")
    print("  EMG2Pose: 0 wrist DoF (20 total)")
    print("  EgoEmg:   2 wrist DoF — F/E + R/U deviation (22 total)")

    # 4. Gesture Diversity
    print()
    print("=" * 64)
    print("4. GESTURE DIVERSITY METRICS")
    print("=" * 64)

    # 4a. Inter-Gesture/Slage Pose Separation
    print("\n4a. Inter-Gesture/Stage Pose Separation")
    print("  Higher mean pairwise distance = gestures occupy more distinct regions")
    if "error" not in sep_emg and "error" not in sep_ego:
        print_table(
            ["Metric", "EMG2Pose (29 stages)", "EgoEmg (60 gestures)"],
            [
                ("Classes used", str(sep_emg["n_classes"]), str(sep_ego["n_classes"])),
                ("Classes available", str(sep_emg["n_classes_total"]), str(sep_ego["n_classes_total"])),
                (
                    "Mean pairwise distance",
                    f"{sep_emg['mean_pairwise_distance']:.4f}",
                    f"{sep_ego['mean_pairwise_distance']:.4f}",
                ),
                (
                    "Min pairwise distance",
                    f"{sep_emg['min_pairwise_distance']:.4f}",
                    f"{sep_ego['min_pairwise_distance']:.4f}",
                ),
                (
                    "Max pairwise distance",
                    f"{sep_emg['max_pairwise_distance']:.4f}",
                    f"{sep_ego['max_pairwise_distance']:.4f}",
                ),
                (
                    "Median pairwise distance",
                    f"{sep_emg['median_pairwise_distance']:.4f}",
                    f"{sep_ego['median_pairwise_distance']:.4f}",
                ),
                (
                    "Std of pairwise distances",
                    f"{sep_emg['std_pairwise_distance']:.4f}",
                    f"{sep_ego['std_pairwise_distance']:.4f}",
                ),
            ],
        )
        ratio_sep = sep_ego["mean_pairwise_distance"] / sep_emg["mean_pairwise_distance"] if sep_emg["mean_pairwise_distance"] > 0 else 0
        print(f"  → EgoEmg/EgoEmg mean separation ratio: {ratio_sep:.2f}x")
    else:
        print(f"  EMG2Pose: {sep_emg.get('error', 'OK')}")
        print(f"  EgoEmg:   {sep_ego.get('error', 'OK')}")

    # 4b. Intra-Gesture Variance Ratio
    print("\n4b. Intra-Gesture Variance Explained")
    print("  Higher variance-explained-by-class = gesture identity carries more pose information")
    if "error" not in var_emg and "error" not in var_ego:
        print_table(
            ["Metric", "EMG2Pose", "EgoEmg"],
            [
                ("Classes used", str(var_emg["n_classes_used"]), str(var_ego["n_classes_used"])),
                (
                    "Mean within-class variance",
                    f"{var_emg['mean_within_class_variance']:.4f}",
                    f"{var_ego['mean_within_class_variance']:.4f}",
                ),
                (
                    "Total variance (class frames)",
                    f"{var_emg['total_variance']:.4f}",
                    f"{var_ego['total_variance']:.4f}",
                ),
                (
                    "Variance explained by class",
                    f"{var_emg['variance_explained_by_class']:.2%}",
                    f"{var_ego['variance_explained_by_class']:.2%}",
                ),
            ],
        )
        if var_ego["variance_explained_by_class"] > var_emg["variance_explained_by_class"]:
            print("  → EgoEmg gesture identity explains MORE pose variance (✓)")
        else:
            print("  → EMG2Pose stage identity explains more (unexpected, investigate)")
    else:
        print(f"  EMG2Pose: {var_emg.get('error', 'OK')}")
        print(f"  EgoEmg:   {var_ego.get('error', 'OK')}")

    # 3. Smoothness
    print()
    print("=" * 64)
    print("3. LABEL SMOOTHNESS (contiguous-block analysis)")
    print("=" * 64)

    if "error" in smooth_emg or "error" in smooth_ego:
        print(f"  Error: EMG2Pose={smooth_emg.get('error')}, EgoEmg={smooth_ego.get('error')}")
    else:
        print(f"\n  Blocks analyzed: EMG2Pose={smooth_emg['n_blocks']}, EgoEmg={smooth_ego['n_blocks']}")
        print(f"  Total frames in blocks: EMG2Pose={smooth_emg['total_frames']:,}, EgoEmg={smooth_ego['total_frames']:,}")
        print(f"  Consecutive valid pairs: EMG2Pose={smooth_emg['total_valid_pairs']:,}, EgoEmg={smooth_ego['total_valid_pairs']:,}")

        print("\n3a. Native-rate outlier rates (distribution-relative, within each block)")
        print_table(
            ["Metric", "EMG2Pose", "EgoEmg", "Note"],
            [
                (
                    "1st-diff outlier rate",
                    f"{smooth_emg['step_outlier_rate']:.4f}",
                    f"{smooth_ego['step_outlier_rate']:.4f}",
                    "Spike frames (lower=cleaner)",
                ),
                (
                    "2nd-diff outlier rate",
                    f"{smooth_emg['acc_outlier_rate']:.4f}",
                    f"{smooth_ego['acc_outlier_rate']:.4f}",
                    "Acceleration spikes",
                ),
                (
                    "3rd-diff outlier rate",
                    f"{smooth_emg['jerk_outlier_rate']:.4f}",
                    f"{smooth_ego['jerk_outlier_rate']:.4f}",
                    "Jerk spikes (lower=cleaner)",
                ),
                (
                    "Zero-change frame rate",
                    f"{smooth_emg['zero_change_rate']:.4f}",
                    f"{smooth_ego['zero_change_rate']:.4f}",
                    "Stale labels (higher=less informative)",
                ),
            ],
        )

        print("\n3b. Native-rate mean absolute change")
        print_table(
            ["Metric", "EMG2Pose", "EgoEmg", "Ratio"],
            [
                (
                    "Mean abs 1st diff (rad/frame)",
                    f"{smooth_emg['mean_abs_step']:.6f}",
                    f"{smooth_ego['mean_abs_step']:.6f}",
                    f"{smooth_ego['mean_abs_step'] / smooth_emg['mean_abs_step']:.2f}x"
                    if smooth_emg.get("mean_abs_step", 0) > 0 else "N/A",
                ),
                (
                    "Mean abs 3rd diff",
                    f"{smooth_emg['mean_abs_3rd_diff']:.6f}",
                    f"{smooth_ego['mean_abs_3rd_diff']:.6f}",
                    f"{smooth_ego['mean_abs_3rd_diff'] / smooth_emg['mean_abs_3rd_diff']:.2f}x"
                    if smooth_emg.get("mean_abs_3rd_diff", 0) > 0 else "N/A",
                ),
            ],
        )

        # Note: decimated metrics are unreliable due to differing label
        # update rates; the outlier analysis above is the fair comparison.

    # 5. Wrist Articulation Coverage (EgoEmg only)
    print()
    print("=" * 64)
    print("5. WRIST ARTICULATION COVERAGE (EgoEmg only)")
    print("=" * 64)
    if "error" not in wrist_ego:
        print(f"\n  Frames sampled: {wrist_ego['n_frames']:,}")
        print(f"  Right hand: {wrist_ego['n_right_frames']:,}, Left hand: {wrist_ego['n_left_frames']:,}")
        print()
        print_table(
            ["Metric", "Pitch (F/E)", "Yaw (R/U)"],
            [
                ("Min", f"{wrist_ego['pitch_min_deg']:.1f}°", f"{wrist_ego['yaw_min_deg']:.1f}°"),
                ("Max", f"{wrist_ego['pitch_max_deg']:.1f}°", f"{wrist_ego['yaw_max_deg']:.1f}°"),
                ("Range", f"{wrist_ego['pitch_range_deg']:.1f}°", f"{wrist_ego['yaw_range_deg']:.1f}°"),
                ("Mean ± Std", f"{wrist_ego['pitch_mean_deg']:.1f} ± {wrist_ego['pitch_std_deg']:.1f}°",
                               f"{wrist_ego['yaw_mean_deg']:.1f} ± {wrist_ego['yaw_std_deg']:.1f}°"),
            ],
        )
        print(f"\n  2D Convex Hull Area: {wrist_ego['convex_hull_area_rad2']:.4f} rad²")
        print("\n  EMG2Pose has 0 wrist DoF — no comparison possible.")
        print("  EgoEmg uniquely provides wrist orientation labels (pitch + yaw).")
        print("  Note: pitch/yaw represent absolute wrist rigid-body orientation,")
        print("  not isolated joint angles. Full 360° range includes forearm rotation.")
    else:
        print(f"  Error: {wrist_ego['error']}")

    # === Summary ===
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    summary_rows = [
        ("IK failure rate", f"{validity_emg['invalid_rate']:.2%}", f"{validity_ego['invalid_rate']:.2%}", "Lower is better"),
        ("Total joint range (rad)", f"{joint_emg['total_range']:.2f}", f"{joint_ego['total_range']:.2f}", "Higher = more diverse"),
        ("PCs for 95% variance", str(pca_emg["n_components_95"]), str(pca_ego["n_components_95"]), "Higher = richer"),
        ("Wrist DoF", "0", "2", "Unique to EgoEmg"),
    ]
    if "error" not in sep_emg and "error" not in sep_ego:
        summary_rows.append((
            "Inter-class pose separation",
            f"{sep_emg['mean_pairwise_distance']:.4f}",
            f"{sep_ego['mean_pairwise_distance']:.4f}",
            "Higher = more distinct gestures",
        ))
    if "error" not in var_emg and "error" not in var_ego:
        summary_rows.append((
            "Variance explained by class",
            f"{var_emg['variance_explained_by_class']:.2%}",
            f"{var_ego['variance_explained_by_class']:.2%}",
            "Higher = classes more informative",
        ))
    if "error" not in smooth_emg and "error" not in smooth_ego:
        summary_rows += [
            ("1st-diff outlier rate", f"{smooth_emg['step_outlier_rate']:.4f}", f"{smooth_ego['step_outlier_rate']:.4f}", "Lower = less spike noise"),
            ("3rd-diff outlier rate", f"{smooth_emg['jerk_outlier_rate']:.4f}", f"{smooth_ego['jerk_outlier_rate']:.4f}", "Lower = smoother"),
        ]
    if "error" not in wrist_ego:
        summary_rows.append((
            "Wrist orientation labels",
            "0 DoF",
            "2 DoF (pitch + yaw)",
            "Unique to EgoEmg",
        ))
    print_table(["Metric", "EMG2Pose", "EgoEmg", "Note"], summary_rows)


if __name__ == "__main__":
    main()
