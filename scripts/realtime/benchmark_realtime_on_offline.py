#!/usr/bin/env python3
"""Benchmark: validate scripts/realtime inference on Manus offline data.

Runs four evaluation modes on the same Manus memmap data and compares MAE:

  1. Offline reference  — standard DataLoader + module.forward() (gold standard)
  2. Full realtime       — raw µV → mV → FFT filter → channel map → normalize → model
  3. Prefiltered + ring  — pre-filtered mV → ring interp → normalize → model
  4. Prefiltered + zero  — pre-filtered mV → zero-pad  → normalize → model

Decomposes the total delta into filter / channel-map / temporal components.

Usage:
    python scripts/realtime/benchmark_realtime_on_offline.py
    python scripts/realtime/benchmark_realtime_on_offline.py --checkpoint /path/to/ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from egoemg.datasets.layout_utils import (
    get_sparse_ring_interp_matrix,
    place_sparse_channels,
)
from egoemg.lightning import EmgPredictionModule

from realtime.inference import (
    InferenceEngine,
    ModelLoader,
    _HW_TO_16CH,
    _RING_INTERP_MATRIX,
)

# ── Defaults ──
DEFAULT_CHECKPOINT = (
    "logs/2026-05-26/01-08-14_emg2pose/regression_manus_finetune/"
    "version_0/checkpoints/manus-small-ft-epoch=049-val_mae=0.1592.ckpt"
)
DEFAULT_MEMMAP_DIR = "data/manus_memmap"
NORM_STATS_PATH = "assets/per_dataset_norm_stats.json"

# Model config (from checkpoint hparams)
WINDOW_LENGTH = 7790
STRIDE = 7790  # non-overlapping, matches val_test_stride
N_ANGLE_DIMS = 20  # after trimming wrist dims (ignore_head_tail_dims=2)
OUT_CHANNELS = 22  # model output channels
CHANNEL_INDICES = [10, 12, 0, 1, 2, 4, 5, 6]

# Norm stats for egoemg (matching training config: dataset_name="egoemg")
NORM_MEAN = 7e-05
NORM_STD = 3.173


# ────────────────────────────────────────────────────────────────────
#  Utilities
# ────────────────────────────────────────────────────────────────────


def load_memmaps(memmap_dir: Path):
    """Open memmap files for direct access."""
    manifest_path = memmap_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    total_rows = manifest["total_rows"]

    raw_emg = np.memmap(
        memmap_dir / "emg_right_raw.dat",
        dtype=np.float32,
        mode="r",
        shape=(total_rows, 8),
    )
    filtered_emg = np.memmap(
        memmap_dir / "emg_right_filtered.dat",
        dtype=np.float32,
        mode="r",
        shape=(total_rows, 8),
    )
    joint_angles = np.memmap(
        memmap_dir / "generated_joint_angles_right.dat",
        dtype=np.float32,
        mode="r",
        shape=(total_rows, N_ANGLE_DIMS),
    )
    label_valid = np.memmap(
        memmap_dir / "generated_label_valid.dat",
        dtype=np.bool_,
        mode="r",
        shape=(total_rows, 2),
    )

    # Episode boundaries from metadata.npz
    meta = np.load(memmap_dir / "metadata.npz", allow_pickle=True)
    ep_starts = meta["episode_start_idx"].tolist()
    ep_ends = meta["episode_end_idx"].tolist()  # inclusive

    return raw_emg, filtered_emg, joint_angles, label_valid, ep_starts, ep_ends


def enumerate_windows(ep_starts, ep_ends, window_length, stride):
    """Yield (global_start, global_end, episode_idx) for non-overlapping windows."""
    windows = []
    for ep_idx, (ep_s, ep_e) in enumerate(zip(ep_starts, ep_ends)):
        ep_len = ep_e - ep_s + 1  # inclusive end → length
        n_win = (ep_len - window_length) // stride + 1
        for i in range(n_win):
            start = ep_s + i * stride
            end = start + window_length  # exclusive
            windows.append((start, end, ep_idx))
    return windows


def reset_engine_filter(engine: InferenceEngine):
    """Reset the InferenceEngine's overlap-save filter history."""
    engine._filter_buf = np.zeros((0, 8), dtype=np.float32)


# ────────────────────────────────────────────────────────────────────
#  Mode 0: Offline reference (DataLoader + module.forward)
# ────────────────────────────────────────────────────────────────────


def run_offline_evaluation(module: EmgPredictionModule, memmap_dir: str, device: str):
    """Standard offline evaluation matching training pipeline exactly."""
    ds = EgoEmgMemmapDataset(
        memmap_dir=memmap_dir,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
        allowed_splits=["train"],
        modalities=["emg", "joint_angles", "labels"],
        target_hand="right",
        emg_field_preference="filtered",
        emg_layout="emg2pose_interpolate16",
        emg2pose_channel_indices=CHANNEL_INDICES,
        channel_interpolate=False,  # matches training
        jitter=False,
        dataset_name="egoemg",  # matches training → egoemg norm stats
        norm_mode="per-dataset",
        norm_stats_path=str(_PROJECT_ROOT / NORM_STATS_PATH),
        center_target_only=False,
    )

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=_offline_collate,
    )

    total_abs = 0.0
    total_n = 0
    for batch in tqdm(loader, desc="Offline eval", unit="batch"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            preds, targets, mask = module.forward(batch)
        # preds/targets: (B, C, T), mask: (B, T)
        C = preds.shape[1]
        abs_per_time = (preds - targets).abs().sum(dim=1)  # (B, T)
        sample_abs = (abs_per_time * mask.float()).sum(dim=1)
        sample_n = mask.float().sum(dim=1) * C
        total_abs += sample_abs.sum().item()
        total_n += sample_n.sum().item()

    mae = total_abs / total_n if total_n > 0 else float("nan")
    n_windows = len(ds)
    return mae, n_windows


def _offline_collate(batch):
    from torch.utils.data._utils.collate import default_collate

    processed = []
    for s in batch:
        emg = torch.as_tensor(s["emg"], dtype=torch.float32)
        ja = torch.as_tensor(s["joint_angles"], dtype=torch.float32)
        mask = torch.as_tensor(s["label_valid_mask"], dtype=torch.bool)
        processed.append({"emg": emg, "joint_angles": ja, "label_valid_mask": mask})
    return default_collate(processed)


# ────────────────────────────────────────────────────────────────────
#  Mode 1: Full realtime (raw µV → engine.predict)
# ────────────────────────────────────────────────────────────────────


def run_full_realtime(
    raw_emg,
    joint_angles,
    label_valid,
    engine: InferenceEngine,
    windows,
    ep_starts_set,
    raw_is_mv: bool = False,
):
    """Feed raw EMG through InferenceEngine, collecting predictions."""
    predictions = []
    gt_list = []
    valid_list = []

    for start, end, ep_idx in tqdm(windows, desc="Mode 1 (full RT)", unit="win"):
        # Reset filter buffer at episode boundaries
        if start in ep_starts_set:
            reset_engine_filter(engine)

        raw_window = np.asarray(raw_emg[start:end], dtype=np.float32)  # (W, 8)
        angles, _ = engine.predict(raw_window, raw_is_mv=raw_is_mv)  # (22,)

        predictions.append(angles[:N_ANGLE_DIMS])
        # GT at last sample of window
        gt_list.append(np.asarray(joint_angles[end - 1]))
        # Right hand validity (column 1)
        valid_list.append(bool(label_valid[end - 1, 1]))

    return np.array(predictions), np.array(gt_list), np.array(valid_list)


# ────────────────────────────────────────────────────────────────────
#  Mode 2/3: Prefiltered EMG (skip FFT, manual channel map + model)
# ────────────────────────────────────────────────────────────────────


def run_prefiltered(
    filtered_emg,
    joint_angles,
    label_valid,
    model,
    windows,
    device,
    use_ring_interp: bool,
    label: str,
):
    """Run streaming with pre-filtered EMG, bypassing FFT filter."""
    positions = _HW_TO_16CH

    predictions = []
    gt_list = []
    valid_list = []

    for start, end, ep_idx in tqdm(windows, desc=label, unit="win"):
        window_mv = np.asarray(
            filtered_emg[start:end], dtype=np.float32
        )  # (W, 8)

        # Channel mapping
        if use_ring_interp:
            emg_16ch = window_mv @ _RING_INTERP_MATRIX  # (W, 16)
        else:
            emg_16ch = place_sparse_channels(
                window_mv, 16, positions
            )  # (W, 16)

        # Normalize
        emg_norm = (emg_16ch - NORM_MEAN) / (NORM_STD + 1e-6)

        # Model forward: (1, 16, W)
        tensor_in = torch.from_numpy(emg_norm.T).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model({"emg": tensor_in})
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, dict):
            output = output.get("angles", output.get("recon"))

        # output: (1, 22, T_pred) — take last timestep, first 20 dims
        angles = output[0, :N_ANGLE_DIMS, -1].cpu().numpy()
        predictions.append(angles)

        gt_list.append(np.asarray(joint_angles[end - 1]))
        valid_list.append(bool(label_valid[end - 1, 1]))

    return np.array(predictions), np.array(gt_list), np.array(valid_list)


# ────────────────────────────────────────────────────────────────────
#  MAE computation
# ────────────────────────────────────────────────────────────────────


def compute_mae(predictions, gt, valid_mask):
    """MAE over valid windows, averaging over all channels."""
    if valid_mask.sum() == 0:
        return float("nan"), 0
    abs_errors = np.abs(predictions[valid_mask] - gt[valid_mask])
    return float(abs_errors.mean()), int(valid_mask.sum())


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark realtime inference on Manus offline data"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(_PROJECT_ROOT / DEFAULT_CHECKPOINT),
        help="Path to Lightning checkpoint",
    )
    parser.add_argument(
        "--memmap-dir",
        default=str(_PROJECT_ROOT / DEFAULT_MEMMAP_DIR),
        help="Path to Manus memmap directory",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    memmap_dir = Path(args.memmap_dir)
    ckpt_path = Path(args.checkpoint)

    print("=" * 70)
    print("  Realtime-on-Offline Benchmark: Manus Data")
    print("=" * 70)
    print(f"  Checkpoint:    {ckpt_path.name}")
    print(f"  Memmap:        {memmap_dir}")
    print(f"  Window:        {WINDOW_LENGTH} ({WINDOW_LENGTH / 2000:.3f}s @ 2kHz)")
    print(f"  Stride:        {STRIDE} (non-overlapping)")
    print(f"  Device:        {device}")
    print()

    # ── Verify ring interp matrix matches layout_utils ──
    ref_matrix = get_sparse_ring_interp_matrix(_HW_TO_16CH, 16)
    assert np.allclose(_RING_INTERP_MATRIX, ref_matrix), (
        "Ring interp matrix mismatch between realtime and layout_utils!"
    )
    print("  [ok] Ring interpolation matrix verified")

    # ── Load model ──
    print("\nLoading model...")
    t0 = time.time()
    loader = ModelLoader(str(ckpt_path), device=device)
    model, model_config = loader.load()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Variant: {model_config.variant}, in_ch={model_config.in_channels}, "
          f"out_ch={model_config.out_channels}")
    print(f"  Norm: mean={model_config.norm_mean}, std={model_config.norm_std}")
    print(f"  Window: {model_config.window_length}")

    # Also load the full Lightning module for offline evaluation
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    module = EmgPredictionModule.load_from_checkpoint(
        str(ckpt_path),
        module_conf=hp["module_conf"],
        optimizer_conf=hp["optimizer_conf"],
        lr_scheduler_conf=hp["lr_scheduler_conf"],
        loss_weights=hp["loss_weights"],
        datamodule=hp.get("datamodule"),
        map_location="cpu",
    )
    module.eval()
    module.to(device)

    # Build InferenceEngine
    print(f"  Channel interpolate: {model_config.channel_interpolate}")
    rt_chmap = "zero-pad" if not model_config.channel_interpolate else "ring interp"
    engine = InferenceEngine(model, model_config, torch.device(device))

    # ── Load memmaps ──
    print("\nLoading memmaps...")
    raw_emg, filtered_emg, joint_angles, label_valid, ep_starts, ep_ends = (
        load_memmaps(memmap_dir)
    )
    total_rows = raw_emg.shape[0]
    print(f"  Total rows: {total_rows:,}")
    print(f"  Episodes: {len(ep_starts)}")
    for i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        print(f"    ep{i}: [{s}, {e}] ({e - s + 1:,} rows)")

    # ── Enumerate windows ──
    windows = enumerate_windows(ep_starts, ep_ends, WINDOW_LENGTH, STRIDE)
    ep_starts_set = set(ep_starts)
    print(f"  Windows: {len(windows)} total")
    for i, (s, e, ep) in enumerate(windows[:3]):
        print(f"    win{i}: [{s}, {e}) ep{ep}")
    if len(windows) > 3:
        print(f"    ...")

    # ── Run evaluations ──
    results = {}

    # Mode 0: Offline reference
    print("\n" + "─" * 70)
    print("  Mode 0: Offline reference (DataLoader + module.forward)")
    print("─" * 70)
    offline_mae, offline_n = run_offline_evaluation(module, str(memmap_dir), device)
    results["offline"] = offline_mae
    print(f"  → MAE = {offline_mae:.6f} rad ({np.rad2deg(offline_mae):.2f}°), "
          f"{offline_n} windows")

    # Mode 1: Full realtime
    print("\n" + "─" * 70)
    print(f"  Mode 1: Full realtime (raw mV → FFT → {rt_chmap} → norm → model)")
    print("─" * 70)
    preds_rt, gt_rt, valid_rt = run_full_realtime(
        raw_emg, joint_angles, label_valid, engine, windows, ep_starts_set,
        raw_is_mv=True,  # memmap raw is already in mV
    )
    rt_mae, rt_n = compute_mae(preds_rt, gt_rt, valid_rt)
    results["realtime"] = rt_mae
    print(f"  → MAE = {rt_mae:.6f} rad ({np.rad2deg(rt_mae):.2f}°), "
          f"{rt_n} valid windows")

    # Mode 2: Prefiltered + ring interp
    print("\n" + "─" * 70)
    print("  Mode 2: Prefiltered + ring interp (skip FFT, ring interp chmap)")
    print("─" * 70)
    preds_ring, gt_ring, valid_ring = run_prefiltered(
        filtered_emg, joint_angles, label_valid, model, windows, device,
        use_ring_interp=True, label="Mode 2 (ring)",
    )
    ring_mae, ring_n = compute_mae(preds_ring, gt_ring, valid_ring)
    results["ring"] = ring_mae
    print(f"  → MAE = {ring_mae:.6f} rad ({np.rad2deg(ring_mae):.2f}°), "
          f"{ring_n} valid windows")

    # Mode 3: Prefiltered + zero-pad
    print("\n" + "─" * 70)
    print("  Mode 3: Prefiltered + zero-pad (skip FFT, zero-pad chmap)")
    print("─" * 70)
    preds_zero, gt_zero, valid_zero = run_prefiltered(
        filtered_emg, joint_angles, label_valid, model, windows, device,
        use_ring_interp=False, label="Mode 3 (zero)",
    )
    zero_mae, zero_n = compute_mae(preds_zero, gt_zero, valid_zero)
    results["zero"] = zero_mae
    print(f"  → MAE = {zero_mae:.6f} rad ({np.rad2deg(zero_mae):.2f}°), "
          f"{zero_n} valid windows")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    header = f"  {'Pipeline':<36} {'MAE(rad)':>10} {'MAE(deg)':>10} {'Δ(rad)':>10} {'Δ(%)':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    rows = [
        ("Offline (full temporal, zero-pad)", offline_mae),
        (f"Realtime (full pipeline, {rt_chmap})", rt_mae),
        ("Prefiltered + ring interp", ring_mae),
        ("Prefiltered + zero-pad", zero_mae),
    ]
    for name, mae in rows:
        deg = np.rad2deg(mae)
        delta = mae - offline_mae
        pct = (delta / offline_mae * 100) if offline_mae > 0 else float("nan")
        delta_str = f"{delta:+.6f}" if not np.isnan(delta) else "  —"
        pct_str = f"{pct:+.1f}%" if not np.isnan(pct) else "  —"
        print(f"  {name:<36} {mae:>10.6f} {deg:>9.2f}° {delta_str:>10} {pct_str:>8}")

    # ── Discrepancy decomposition ──
    print("\n" + "=" * 70)
    print("  DISCREPANCY DECOMPOSITION")
    print("=" * 70)

    filter_impact = rt_mae - ring_mae
    chmap_impact = ring_mae - zero_mae
    temporal_impact = zero_mae - offline_mae
    total_delta = rt_mae - offline_mae

    print(f"  FFT filter (per-window vs full-session):  {filter_impact:+.6f} rad "
          f"({np.rad2deg(filter_impact):+.3f}°)")
    print(f"  Channel map (ring interp vs zero-pad):    {chmap_impact:+.6f} rad "
          f"({np.rad2deg(chmap_impact):+.3f}°)")
    print(f"  Temporal (last-step vs full-temporal):    {temporal_impact:+.6f} rad "
          f"({np.rad2deg(temporal_impact):+.3f}°)")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Total delta (RT - offline):               {total_delta:+.6f} rad "
          f"({np.rad2deg(total_delta):+.3f}°)")
    print(f"  Sum of components:                        "
          f"{filter_impact + chmap_impact + temporal_impact:+.6f} rad")

    # ── Per-window stats ──
    print("\n" + "=" * 70)
    print("  PER-WINDOW STATISTICS (Realtime)")
    print("=" * 70)
    per_win_mae = np.abs(preds_rt[valid_rt] - gt_rt[valid_rt]).mean(axis=1)
    print(f"  Mean:   {per_win_mae.mean():.6f} rad ({np.rad2deg(per_win_mae.mean()):.2f}°)")
    print(f"  Std:    {per_win_mae.std():.6f} rad")
    print(f"  Min:    {per_win_mae.min():.6f} rad (window #{per_win_mae.argmin()})")
    print(f"  Max:    {per_win_mae.max():.6f} rad (window #{per_win_mae.argmax()})")
    print(f"  Median: {np.median(per_win_mae):.6f} rad")

    # ── Recommendations ──
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    rt_uses_zero_pad = not model_config.channel_interpolate
    ring_impact_on_this_model = ring_mae - zero_mae  # what ring interp WOULD cost

    if rt_uses_zero_pad and abs(ring_impact_on_this_model) > 0.01:
        print(f"  ✓ Channel mapping: engine uses zero-pad (matches training).")
        print(f"    Ring interp would add +{np.rad2deg(ring_impact_on_this_model):.1f}° — correctly avoided.")
    elif not rt_uses_zero_pad and abs(ring_impact_on_this_model) > 0.01:
        print(f"  ⚠ Channel mapping MISMATCH: engine uses ring interp but training")
        print(f"    used zero-pad. This adds +{np.rad2deg(ring_impact_on_this_model):.1f}° error.")
        print(f"    Set channel_interpolate=False in ModelLoader or ServerConfig.")

    if abs(total_delta) < 0.02:
        print("  ✓ Realtime MAE is within 0.02 rad of offline — pipeline is aligned.")
    elif abs(total_delta) < 0.05:
        remaining_src = "per-window FFT vs full-session filter"
        print(f"  ~ Realtime MAE within 0.05 rad of offline — acceptable.")
        print(f"    Remaining {np.rad2deg(abs(total_delta)):.1f}° gap is from {remaining_src}.")
    else:
        print("  ✗ Realtime MAE differs significantly from offline — investigate above.")

    print()


if __name__ == "__main__":
    main()
