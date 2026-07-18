#!/usr/bin/env python3
"""Parse MANO generation log and produce a summary report."""

import re
import sys
from pathlib import Path

import numpy as np


def parse_log(log_path: str) -> list[dict]:
    pattern = re.compile(
        r"\[(\d+)/(\d+)\]\s+(\S+):\s+([\d,]+)fr\s+->\s+([\d,]+)inf\s+\|"
        r"\s+GPU=([\d.]+)s.*?"
        r"Lerr=([\d.]+)mm\s+Rerr=([\d.]+)mm"
    )
    episodes = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episodes.append({
                    "idx": int(m.group(1)),
                    "total": int(m.group(2)),
                    "name": m.group(3),
                    "frames": int(m.group(4).replace(",", "")),
                    "inferred": int(m.group(5).replace(",", "")),
                    "gpu_sec": float(m.group(6)),
                    "left_err_mm": float(m.group(7)),
                    "right_err_mm": float(m.group(8)),
                })
    return episodes


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mano_full_gen_v2.log"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/EgoEMG/mano/chunk-000/generation_report.txt"

    episodes = parse_log(log_path)
    if not episodes:
        print("No episodes found in log.")
        return

    left_errs = [e["left_err_mm"] for e in episodes]
    right_errs = [e["right_err_mm"] for e in episodes]
    avg_errs = [(l + r) / 2 for l, r in zip(left_errs, right_errs)]
    total_frames = sum(e["frames"] for e in episodes)
    total_gpu = sum(e["gpu_sec"] for e in episodes)

    lines = []
    lines.append("=" * 90)
    lines.append("MANO Parameter Generation Report")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Checkpoint: v51_best_ep12 (m2m_pose_shape_run/version_51)")
    lines.append(f"  Path: tb_logs/m2m_pose_shape_run/version_51/checkpoints/"
                 f"best-model-epoch=12-val/loss_total=0.0002.ckpt")
    lines.append(f"Model: EfficientGraphTransformer (hidden=1280, layers=12, heads=8)")
    lines.append(f"Inference stride: 100, batch_size: 2000")
    lines.append("Left-hand strategy: raw-MANO mirror_x before world alignment")
    lines.append("")
    lines.append(f"Episodes processed: {len(episodes)}")
    lines.append(f"Total frames: {total_frames:,}")
    lines.append(f"Total GPU time: {total_gpu:.1f}s ({total_gpu/60:.1f}min)")
    lines.append(f"Avg GPU time per episode: {total_gpu/len(episodes):.1f}s")
    lines.append("")
    lines.append("-" * 90)
    lines.append(f"{'Episode':<25} {'Frames':>12} {'GPU(s)':>8} {'Left(mm)':>10} {'Right(mm)':>10} {'Avg(mm)':>10}")
    lines.append("-" * 90)
    for e in episodes:
        avg = (e["left_err_mm"] + e["right_err_mm"]) / 2
        lines.append(
            f"{e['name']:<25} {e['frames']:>12,} {e['gpu_sec']:>8.1f} "
            f"{e['left_err_mm']:>10.1f} {e['right_err_mm']:>10.1f} {avg:>10.1f}"
        )
    lines.append("-" * 90)
    lines.append("")
    lines.append("Summary Statistics (Kabsch-aligned marker error, mm):")
    lines.append(f"  Left hand:  mean={np.mean(left_errs):.2f}, std={np.std(left_errs):.2f}, "
                 f"min={np.min(left_errs):.2f}, max={np.max(left_errs):.2f}")
    lines.append(f"  Right hand: mean={np.mean(right_errs):.2f}, std={np.std(right_errs):.2f}, "
                 f"min={np.min(right_errs):.2f}, max={np.max(right_errs):.2f}")
    lines.append(f"  Average:    mean={np.mean(avg_errs):.2f}, std={np.std(avg_errs):.2f}, "
                 f"min={np.min(avg_errs):.2f}, max={np.max(avg_errs):.2f}")
    lines.append("")
    lines.append("Output files per episode (data/EgoEMG/mano/chunk-000/):")
    lines.append("  {episode_id}_{hand}_pose.npy  — (T, 48) float32, MANO axis-angle")
    lines.append("  {episode_id}_{hand}_beta.npy  — (10,) float32, MANO shape")
    lines.append("  {episode_id}_{hand}_trans.npy — (3,) float32, mean Kabsch translation")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Both left and right pose are stored in MANO-RIGHT canonical parameterization")
    lines.append("  - Left-hand world alignment uses raw-MANO mirror_x before Kabsch")
    lines.append("  - Pose[:, :3] (global rotation) is zeroed; local frame only")
    lines.append("  - Kabsch error measures post-alignment residual between predicted")
    lines.append("    MANO surface markers and GT mocap keypoints (21 markers)")
    lines.append("=" * 90)

    report = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
