#!/usr/bin/env python3
"""
Improved pose-space diversity comparison between EgoEmg and EMG2Pose.

Metrics (all computed in a shared PCA space fit on the union of both datasets):
1. PCA pose-space coverage visualization (PC1 vs PC2 scatter)
2. Occupancy score + Entropy in PCA bin grid (with bootstrap 95% CI)
3. Effective dimensionality (participation ratio of eigenvalue spectrum)
4. Per-joint robust range (P95 - P5)

Key design decisions:
- PCA is fit on the UNION of both datasets (not separately)
- Equal number of frames sampled from each dataset
- Bootstrap 95% CI for occupancy and entropy
- Robust range (P5-P95) instead of min-max for per-joint analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

# Don't require sklearn at import time — it's slow to load
PCA = None


def _get_pca():
    global PCA
    if PCA is None:
        from sklearn.decomposition import PCA as _PCA
        PCA = _PCA
    return PCA


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


def _open_memmap(mm_dir, info):
    return np.memmap(
        mm_dir / info["filename"],
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def load_emg2pose(memmap_dir):
    mm_dir = Path(memmap_dir)
    manifest = json.loads((mm_dir / "manifest.json").read_text())
    fields = manifest["fields"]
    ja = _open_memmap(mm_dir, fields["joint_angles"])
    vm = _open_memmap(mm_dir, fields["valid_mask"])
    return ja, vm


def load_egoemg(memmap_dir):
    mm_dir = Path(memmap_dir)
    manifest = json.loads((mm_dir / "manifest.json").read_text())
    fields = manifest["fields"]
    ja_l = _open_memmap(mm_dir, fields["generated_joint_angles_left"])
    ja_r = _open_memmap(mm_dir, fields["generated_joint_angles_right"])
    vm = _open_memmap(mm_dir, fields["generated_label_valid"])
    return ja_l, ja_r, vm


def sample_valid_frames(ja_arrays, vm_arrays, n_frames, seed):
    """Sample n_frames valid frames from memmaps.

    Args:
        ja_arrays: list of (N_i, 20) memmaps
        vm_arrays: list of (N_i,) memmaps (or (N_i, 2) for EgoEmg with hand mask)
    Returns:
        sampled_ja: (n_frames, 20) float32
    """
    rng = np.random.RandomState(seed)

    # Flatten to find all valid indices
    offsets = [0]
    valid_indices = []
    for ja_mm, vm_mm in zip(ja_arrays, vm_arrays):
        if vm_mm.ndim == 2:
            v = vm_mm[:, 0] | vm_mm[:, 1]  # either hand valid
        else:
            v = vm_mm
        # Find valid indices within this memmap
        local_valid = np.where(v)[0]
        # Map to global index space
        global_valid = local_valid + offsets[-1]
        valid_indices.append(global_valid)
        offsets.append(offsets[-1] + len(ja_mm))

    all_valid = np.concatenate(valid_indices)
    rng.shuffle(all_valid)
    chosen = np.sort(all_valid[:n_frames])

    # Extract frames
    chunks = []
    for ja_mm, off_start, off_end in zip(ja_arrays, offsets[:-1], offsets[1:]):
        local = chosen[(chosen >= off_start) & (chosen < off_end)] - off_start
        if len(local) > 0:
            chunks.append(np.array(ja_mm[local]))
    return np.concatenate(chunks, axis=0)


def effective_dimensionality(eigenvalues):
    """Participation ratio: D_eff = (Σ λ_i)² / Σ λ_i²"""
    s = eigenvalues.sum()
    if s == 0:
        return 0.0
    return float(s ** 2 / (eigenvalues ** 2).sum())


def compute_occupancy_entropy(pca_embedding, n_bins_per_dim=10, eps=1e-10):
    """Compute occupancy score and normalized entropy in PCA bin grid.

    Args:
        pca_embedding: (N, K) — PCA-projected data
        n_bins_per_dim: int — number of bins per PCA dimension
        eps: small constant for log(0) avoidance

    Returns:
        occupancy: float — fraction of bins occupied
        entropy_norm: float — normalized entropy H / log(n_occupied)
        entropy_raw: float — raw entropy in nats
        n_occupied: int — number of occupied bins
        total_bins: int — total number of bins in grid
    """
    N, K = pca_embedding.shape
    total_bins = n_bins_per_dim ** K

    # Discretize each dimension into bins
    bin_indices = np.zeros((N, K), dtype=np.int32)
    for k in range(K):
        col = pca_embedding[:, k]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin < 1e-10:
            bin_indices[:, k] = 0
        else:
            bin_idx = np.floor((col - cmin) / (cmax - cmin) * n_bins_per_dim)
            bin_idx = np.clip(bin_idx, 0, n_bins_per_dim - 1)
            bin_indices[:, k] = bin_idx.astype(np.int32)

    # Flatten to 1D bin index
    multipliers = n_bins_per_dim ** np.arange(K)
    flat_bins = (bin_indices * multipliers).sum(axis=1)

    unique_bins, counts = np.unique(flat_bins, return_counts=True)
    n_occupied = len(unique_bins)
    occupancy = n_occupied / total_bins

    # Entropy
    probs = counts / counts.sum()
    entropy_raw = float(-np.sum(probs * np.log(probs + eps)))
    entropy_norm = entropy_raw / np.log(n_occupied) if n_occupied > 1 else 0.0

    return {
        "occupancy": float(occupancy),
        "entropy_raw": entropy_raw,
        "entropy_norm": float(entropy_norm),
        "n_occupied": int(n_occupied),
        "total_bins": int(total_bins),
    }


def bootstrap_occupancy_entropy(pca_embedding, n_bins_per_dim=10, n_bootstrap=1000, seed=42):
    """Bootstrap occupancy and entropy with 95% CI."""
    rng = np.random.RandomState(seed)
    N = len(pca_embedding)
    occ_samples = []
    ent_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        result = compute_occupancy_entropy(pca_embedding[idx], n_bins_per_dim)
        occ_samples.append(result["occupancy"])
        ent_samples.append(result["entropy_norm"])

    occ_samples = np.array(occ_samples)
    ent_samples = np.array(ent_samples)

    def ci95(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "occupancy_mean": float(occ_samples.mean()),
        "occupancy_ci95": ci95(occ_samples),
        "entropy_mean": float(ent_samples.mean()),
        "entropy_ci95": ci95(ent_samples),
    }


def compute_metrics(ja_a, ja_b, n_pca_dims=8, n_bins_per_dim=8, seed=42):
    """Compute all comparison metrics between two datasets.

    Args:
        ja_a: (N, 20) — dataset A (EMG2Pose)
        ja_b: (N, 20) — dataset B (EgoEmg)
        n_pca_dims: PCA dimensions to use for binning
        n_bins_per_dim: bins per PCA dimension
    """
    N_a, N_b = len(ja_a), len(ja_b)
    assert N_a == N_b, f"Must have equal samples: {N_a} vs {N_b}"

    # ── Shared PCA on union ──
    X_union = np.concatenate([ja_a, ja_b], axis=0)
    # Standardize
    mu = X_union.mean(axis=0, keepdims=True)
    sigma = X_union.std(axis=0, keepdims=True).clip(min=1e-10)
    X_norm = (X_union - mu) / sigma

    PCA_cls = _get_pca()
    pca = PCA_cls(n_components=min(20, X_norm.shape[1]))
    Z = pca.fit_transform(X_norm)  # (2N, 20)
    Z_a = Z[:N_a]  # first N_a rows = EMG2Pose
    Z_b = Z[N_a:]  # last N_b rows = EgoEmg

    # ── 1. PCA explained variance ──
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_pcs_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    n_pcs_90 = int(np.searchsorted(cumsum, 0.90) + 1)

    # ── 2. Occupancy & Entropy (using K=n_pca_dims PCs) ──
    K = min(n_pca_dims, Z.shape[1])
    Zk_a = Z_a[:, :K]
    Zk_b = Z_b[:, :K]

    base_result_a = compute_occupancy_entropy(Zk_a, n_bins_per_dim)
    base_result_b = compute_occupancy_entropy(Zk_b, n_bins_per_dim)

    bs_a = bootstrap_occupancy_entropy(Zk_a, n_bins_per_dim, seed=seed)
    bs_b = bootstrap_occupancy_entropy(Zk_b, n_bins_per_dim, seed=seed)

    # ── 3. Effective dimensionality ──
    cov_a = np.cov(ja_a, rowvar=False)
    cov_b = np.cov(ja_b, rowvar=False)
    eig_a = np.linalg.eigvalsh(cov_a)[::-1]
    eig_b = np.linalg.eigvalsh(cov_b)[::-1]
    d_eff_a = effective_dimensionality(eig_a)
    d_eff_b = effective_dimensionality(eig_b)

    # ── 4. Per-joint robust range (P5-P95) ──
    def robust_range(ja):
        p5 = np.percentile(ja, 5, axis=0)
        p95 = np.percentile(ja, 95, axis=0)
        return p95 - p5

    rr_a = robust_range(ja_a)
    rr_b = robust_range(ja_b)

    robust_range_by_finger = {}
    for fg_name, indices in FINGER_GROUPS.items():
        robust_range_by_finger[fg_name] = {
            "emg2pose": float(rr_a[indices].sum()),
            "egoemg": float(rr_b[indices].sum()),
        }

    return {
        "n_samples_per_dataset": N_a,
        "n_pca_dims_for_binning": K,
        "n_bins_per_dim": n_bins_per_dim,
        "pca": {
            "n_pcs_90": n_pcs_90,
            "n_pcs_95": n_pcs_95,
            "explained_variance_top10": pca.explained_variance_ratio_[:10].tolist(),
        },
        "occupancy_entropy": {
            "emg2pose": {**base_result_a, **bs_a},
            "egoemg": {**base_result_b, **bs_b},
        },
        "effective_dimensionality": {
            "emg2pose": d_eff_a,
            "egoemg": d_eff_b,
            "ratio": d_eff_b / d_eff_a if d_eff_a > 0 else float("nan"),
        },
        "robust_range": {
            "emg2pose": {name: float(rr_a[i]) for i, name in enumerate(JOINT_NAMES)},
            "egoemg": {name: float(rr_b[i]) for i, name in enumerate(JOINT_NAMES)},
            "emg2pose_total": float(rr_a.sum()),
            "egoemg_total": float(rr_b.sum()),
            "ratio_total": float(rr_b.sum() / rr_a.sum()) if rr_a.sum() > 0 else float("nan"),
        },
        "robust_range_by_finger": robust_range_by_finger,
    }


def print_results(metrics):
    """Pretty-print the comparison metrics."""
    oe = metrics["occupancy_entropy"]
    rr = metrics["robust_range"]
    rrf = metrics["robust_range_by_finger"]

    print(f"\n{'='*70}")
    print(f"POSE-SPACE DIVERSITY COMPARISON")
    print(f"  Samples per dataset: {metrics['n_samples_per_dataset']:,}")
    print(f"  PCA binning: {metrics['n_pca_dims_for_binning']}D × {metrics['n_bins_per_dim']} bins/dim")
    print(f"  Total bins: {oe['egoemg']['total_bins']:,}")
    print(f"{'='*70}")

    # PCA
    pca = metrics["pca"]
    print(f"\n1. SHARED PCA (fit on union)")
    print(f"   PCs for 90% variance: {pca['n_pcs_90']}")
    print(f"   PCs for 95% variance: {pca['n_pcs_95']}")
    print(f"   Top-5 explained var: {', '.join(f'{v:.3f}' for v in pca['explained_variance_top10'][:5])}")

    # Occupancy & Entropy
    print(f"\n2. POSE-SPACE OCCUPANCY & ENTROPY")
    print(f"   {'Metric':<30s} {'EMG2Pose':>20s} {'EgoEmg':>20s}")
    print(f"   {'-'*70}")
    a, b = oe["emg2pose"], oe["egoemg"]
    print(f"   {'Occupied bins':<30s} {a['n_occupied']:>20,} {b['n_occupied']:>20,}")
    print(f"   {'Coverage score':<30s} {a['occupancy']:>19.4f}  {b['occupancy']:>19.4f}")
    print(f"   {'Coverage 95% CI':<30s} [{a['occupancy_ci95'][0]:.4f}, {a['occupancy_ci95'][1]:.4f}]  [{b['occupancy_ci95'][0]:.4f}, {b['occupancy_ci95'][1]:.4f}]")
    print(f"   {'Normalized entropy':<30s} {a['entropy_norm']:>19.4f}  {b['entropy_norm']:>19.4f}")
    print(f"   {'Entropy 95% CI':<30s} [{a['entropy_ci95'][0]:.4f}, {a['entropy_ci95'][1]:.4f}]  [{b['entropy_ci95'][0]:.4f}, {b['entropy_ci95'][1]:.4f}]")

    ratio_occ = b["occupancy_mean"] / a["occupancy_mean"] if a["occupancy_mean"] > 0 else 0
    ratio_ent = b["entropy_mean"] / a["entropy_mean"] if a["entropy_mean"] > 0 else 0
    print(f"\n   Coverage ratio (EgoEmg / EMG2Pose): {ratio_occ:.2f}x")
    print(f"   Entropy ratio  (EgoEmg / EMG2Pose): {ratio_ent:.2f}x")

    # Effective dimensionality
    ed = metrics["effective_dimensionality"]
    print(f"\n3. EFFECTIVE DIMENSIONALITY")
    print(f"   EMG2Pose: {ed['emg2pose']:.2f}")
    print(f"   EgoEmg:   {ed['egoemg']:.2f}")
    print(f"   Ratio:    {ed['ratio']:.2f}x")

    # Robust range
    print(f"\n4. PER-JOINT ROBUST RANGE (P95-P5, radians)")
    print(f"   {'Joint':<20s} {'EMG2Pose':>10s} {'EgoEmg':>10s} {'Ratio':>8s}")
    print(f"   {'-'*50}")
    for i, name in enumerate(JOINT_NAMES):
        ra = rr["emg2pose"][name]
        rb = rr["egoemg"][name]
        ratio = f"{rb/ra:.2f}x" if ra > 0.001 else "N/A"
        print(f"   {name:<20s} {ra:>10.3f} {rb:>10.3f} {ratio:>8s}")

    # Finger group summary
    print(f"\n   By finger group:")
    for fg_name in FINGER_GROUPS:
        ra = rrf[fg_name]["emg2pose"]
        rb = rrf[fg_name]["egoemg"]
        ratio = f"{rb/ra:.2f}x" if ra > 0.001 else "N/A"
        print(f"   {fg_name:<10s} {ra:>10.3f} {rb:>10.3f} {ratio:>8s}")

    print(f"\n   {'TOTAL':<10s} {rr['emg2pose_total']:>10.2f} {rr['egoemg_total']:>10.2f} {rr['ratio_total']:.2f}x")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emg2pose-dir", required=True, type=str)
    parser.add_argument("--egoemg-dir", required=True, type=str)
    parser.add_argument("--n-frames", type=int, default=20000,
                        help="Number of valid frames to sample PER DATASET")
    parser.add_argument("--n-pca-dims", type=int, default=8,
                        help="PCA dimensions for bin grid")
    parser.add_argument("--n-bins", type=int, default=8,
                        help="Bins per PCA dimension")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading EMG2Pose...", file=sys.stderr)
    ja_emg, vm_emg = load_emg2pose(args.emg2pose_dir)

    print(f"Loading EgoEmg...", file=sys.stderr)
    ja_ego_l, ja_ego_r, vm_ego = load_egoemg(args.egoemg_dir)

    print(f"Sampling {args.n_frames:,} valid frames per dataset...", file=sys.stderr)
    ja_a = sample_valid_frames([ja_emg], [vm_emg], args.n_frames, args.seed)
    ja_b = sample_valid_frames(
        [ja_ego_l, ja_ego_r],
        [vm_ego[:, 0], vm_ego[:, 1]],
        args.n_frames,
        args.seed,
    )

    # Filter any non-finite values
    ja_a = ja_a[np.isfinite(ja_a).all(axis=1)]
    ja_b = ja_b[np.isfinite(ja_b).all(axis=1)]
    # Ensure equal size after filtering
    n_min = min(len(ja_a), len(ja_b))
    ja_a = ja_a[:n_min]
    ja_b = ja_b[:n_min]
    print(f"  After filtering: {n_min:,} frames each", file=sys.stderr)

    metrics = compute_metrics(
        ja_a, ja_b,
        n_pca_dims=args.n_pca_dims,
        n_bins_per_dim=args.n_bins,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_results(metrics)


if __name__ == "__main__":
    main()
