#!/usr/bin/env python3
"""
Generate paper-quality comparison figures for EgoEmg vs EMG2Pose.

Outputs:
    paper/figures/fig_joint_range_comparison.pdf  — 22-joint grouped bar chart
    paper/figures/fig_gesture_tsne.pdf             — t-SNE of per-gesture means

Usage:
    python scripts/paper/plot_paper_figures.py \
        --emg2pose-dir /path/to/emg2pose_memmap \
        --egoemg-dir /path/to/EgoEMG_v2_memmap \
        --output-dir paper/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

JOINT_NAMES = [
    "THUMB\nCMC_FE", "THUMB\nCMC_AA", "THUMB\nMCP_FE", "THUMB\nIP_FE",
    "INDEX\nMCP_AA", "INDEX\nMCP_FE", "INDEX\nPIP_FE", "INDEX\nDIP_FE",
    "MIDDLE\nMCP_AA", "MIDDLE\nMCP_FE", "MIDDLE\nPIP_FE", "MIDDLE\nDIP_FE",
    "RING\nMCP_AA", "RING\nMCP_FE", "RING\nPIP_FE", "RING\nDIP_FE",
    "PINKY\nMCP_AA", "PINKY\nMCP_FE", "PINKY\nPIP_FE", "PINKY\nDIP_FE",
    "WRIST\nF/E", "WRIST\nR/U",
]

JOINT_LABELS_SIMPLE = [
    "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
    "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
    "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
    "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
    "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
    "WRIST_FE", "WRIST_RU",
]

FINGER_GROUPS = {
    "Thumb": (0, 4),
    "Index": (4, 8),
    "Middle": (8, 12),
    "Ring": (12, 16),
    "Pinky": (16, 20),
    "Wrist": (20, 22),
}

COLOR_EMG2POSE = "#E8A87C"   # warm tan
COLOR_EGOEMG = "#5B9BD5"     # blue
COLOR_EMPTY = "#CCCCCC"      # light gray for missing data


# ─── data loading (mirrors compare_dataset_labels.py) ────────────

def _open_memmap(mm_dir, info):
    return np.memmap(
        mm_dir / info["filename"],
        dtype=np.dtype(info["dtype"]),
        mode="r",
        shape=tuple(info["shape"]),
    )


def load_emg2pose_meta(memmap_dir):
    mm_dir = Path(memmap_dir)
    manifest = json.loads((mm_dir / "manifest.json").read_text())
    fields = manifest["fields"]
    meta = np.load(mm_dir / "metadata.npz", allow_pickle=True)
    ja = _open_memmap(mm_dir, fields["joint_angles"])
    vm = _open_memmap(mm_dir, fields["valid_mask"])
    blocks_start = meta["blocks_start"]
    blocks_end = meta["blocks_end"]
    session_stage_id = meta["session_stage_id"]
    session_start = meta["session_start_idx"]
    session_end = meta["session_end_idx"]
    return ja, vm, blocks_start, blocks_end, session_stage_id, session_start, session_end


def load_egoemg_meta(memmap_dir):
    """Load EgoEmg finger joint angles (20D) + wrist angles (2D, stored in degrees)."""
    mm_dir = Path(memmap_dir)
    manifest = json.loads((mm_dir / "manifest.json").read_text())
    fields = manifest["fields"]
    ja_l = _open_memmap(mm_dir, fields["generated_joint_angles_left"])
    ja_r = _open_memmap(mm_dir, fields["generated_joint_angles_right"])
    vm_full = _open_memmap(mm_dir, fields["generated_label_valid"])
    gc = _open_memmap(mm_dir, fields["label_gesture_class"])
    ga = _open_memmap(mm_dir, fields["label_gesture_active"])
    # Wrist angles: separate memmap fields, stored in degrees
    pitch_r = _open_memmap(mm_dir, fields["mocap_right_wrist_pitch"])
    yaw_r = _open_memmap(mm_dir, fields["mocap_right_wrist_yaw"])
    v_wrist_r = _open_memmap(mm_dir, fields["mocap_right_wrist_angles_valid"])
    pitch_l = _open_memmap(mm_dir, fields["mocap_left_wrist_pitch"])
    yaw_l = _open_memmap(mm_dir, fields["mocap_left_wrist_yaw"])
    v_wrist_l = _open_memmap(mm_dir, fields["mocap_left_wrist_angles_valid"])
    return (ja_l, ja_r, vm_full, gc, ga,
            pitch_r, yaw_r, v_wrist_r, pitch_l, yaw_l, v_wrist_l)


def sample_random_frames(memmap_arrays, memmap_masks, max_frames, seed):
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


def sample_wrist_for_frames(global_idx, pitch_mm, yaw_mm, valid_mm, arm_idx):
    """Extract wrist angles (deg→rad) for sampled frames of one arm."""
    arm_valid = valid_mm[:, arm_idx] if valid_mm.ndim == 2 else valid_mm
    global_idx_sorted = np.sort(global_idx)
    idx_in_arm = global_idx_sorted % len(pitch_mm)
    mask = arm_valid[idx_in_arm]
    p = np.deg2rad(np.array(pitch_mm[idx_in_arm[mask]]))
    y = np.deg2rad(np.array(yaw_mm[idx_in_arm[mask]]))
    finite = np.isfinite(p) & np.isfinite(y)
    return p[finite], y[finite]


# ─── Figure 1: Per-joint angular range (22 joints) ───────────────

def plot_joint_range(emg_ranges, ego_ranges, output_path):
    """Grouped bar chart: 22 joints × 2 datasets. EMG2Pose has no wrist data."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    n_joints = len(JOINT_NAMES)
    x = np.arange(n_joints)
    width = 0.35

    bars1 = ax.bar(x - width / 2, emg_ranges, width, label="EMG2Pose",
                   color=COLOR_EMG2POSE, edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + width / 2, ego_ranges, width, label="EgoEmg (ours)",
                   color=COLOR_EGOEMG, edgecolor="white", linewidth=0.3)

    ax.set_ylabel("Angular range (rad)")
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS_SIMPLE, rotation=45, ha="right", fontsize=6.5)
    ax.legend(loc="upper right", framealpha=0.9)

    # Finger group background shading
    for fg_name, (start, end) in FINGER_GROUPS.items():
        ax.axvspan(start - 0.5, end - 0.5, alpha=0.04, color="gray")
        mid = (start + end - 1) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, fg_name, ha="center",
                fontsize=7, fontstyle="italic", color="gray")

    # Ratio annotations on the top few joints (skip wrist where EMG2Pose=0)
    for i, (e, g) in enumerate(zip(emg_ranges, ego_ranges)):
        if e > 0.001 and g / e >= 3.0:
            ax.annotate(f"{g / e:.1f}×", (x[i] + width / 2, g),
                        textcoords="offset points", xytext=(0, 4),
                        fontsize=5.5, ha="center", color=COLOR_EGOEMG)

    # Mark wrist columns: EMG2Pose has no data
    for i in range(20, n_joints):
        ax.annotate("N/A", (x[i] - width / 2, 0.15), ha="center",
                    fontsize=5.5, color=COLOR_EMG2POSE, fontstyle="italic")

    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


# ─── Figure 2: t-SNE of per-gesture mean poses ──────────────────

def plot_gesture_tsne(ego_ja, ego_vm, ego_gc, ego_ga, output_path):
    """t-SNE embedding of per-gesture mean 20D pose vectors."""
    from sklearn.manifold import TSNE

    valid = ego_vm & ego_ga & np.isfinite(ego_ja).all(axis=1)
    unique_classes = np.unique(ego_gc[valid])
    unique_classes = unique_classes[unique_classes >= 0]

    per_class_means = {}
    for cls in unique_classes:
        mask = valid & (ego_gc == cls)
        cls_ja = ego_ja[mask]
        if len(cls_ja) < 20:
            continue
        per_class_means[cls] = cls_ja.mean(axis=0)

    class_ids = sorted(per_class_means.keys())
    means = np.array([per_class_means[c] for c in class_ids])

    tsne = TSNE(n_components=2, perplexity=min(10, len(class_ids) - 1),
                random_state=42, metric="euclidean")
    embedded = tsne.fit_transform(means)

    # Color by finger emphasis: which finger group has largest range per gesture
    def dominant_finger(mean_pose):
        ranges = []
        for fg, (s, e) in FINGER_GROUPS.items():
            if fg == "Wrist":
                continue  # skip wrist for coloring
            ranges.append(mean_pose[s:e].max() - mean_pose[s:e].min())
        return np.argmax(ranges)

    finger_names = [n for n in FINGER_GROUPS.keys() if n != "Wrist"]
    colors = plt.cm.tab10([dominant_finger(per_class_means[c]) for c in class_ids])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(embedded[:, 0], embedded[:, 1], c=colors, s=40,
               edgecolors="white", linewidth=0.5, zorder=3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(i), label=name)
                       for i, name in enumerate(finger_names)]
    ax.legend(handles=legend_elements, title="Dominant finger group",
              fontsize=7, title_fontsize=7, loc="best")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of {len(class_ids)} per-gesture mean poses (EgoEmg)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


# ─── main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper comparison figures")
    parser.add_argument("--emg2pose-dir", required=True, type=str)
    parser.add_argument("--egoemg-dir", required=True, type=str)
    parser.add_argument("--output-dir", default="paper/figures", type=str)
    parser.add_argument("--max-frames", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("Loading EMG2Pose...", file=sys.stderr)
    ja_emg, vm_emg, blk_start, blk_end, sess_stage, sess_start, sess_end = \
        load_emg2pose_meta(args.emg2pose_dir)

    print("Loading EgoEmg...", file=sys.stderr)
    (ja_ego_l, ja_ego_r, vm_ego_full, ego_gc_mm, ego_ga_mm,
     pitch_r, yaw_r, v_wrist_r, pitch_l, yaw_l, v_wrist_l) = load_egoemg_meta(args.egoemg_dir)

    # ── Sample frames ──
    print("Sampling frames...", file=sys.stderr)
    emg_ja_sample, emg_vm_sample, _ = sample_random_frames(
        [ja_emg], [vm_emg], args.max_frames, args.seed)
    ego_ja_sample, ego_vm_sample, ego_global_idx = sample_random_frames(
        [ja_ego_l, ja_ego_r],
        [vm_ego_full[:, 0], vm_ego_full[:, 1]],
        args.max_frames, args.seed,
    )

    # ── Figure 1: 22-joint range comparison ──
    print("Generating joint range figure...", file=sys.stderr)

    # EMG2Pose: 20 finger joints only, pad wrist with 0
    emg_valid = emg_ja_sample[emg_vm_sample & np.isfinite(emg_ja_sample).all(axis=1)]
    emg_finger_ranges = emg_valid.max(axis=0) - emg_valid.min(axis=0)
    emg_ranges_22 = np.concatenate([emg_finger_ranges, np.zeros(2)])

    # EgoEmg: 20 finger joints from ja arrays + 2 wrist joints from mocap fields
    ego_valid_finger = ego_ja_sample[ego_vm_sample & np.isfinite(ego_ja_sample).all(axis=1)]
    ego_finger_ranges = ego_valid_finger.max(axis=0) - ego_valid_finger.min(axis=0)

    # Sample wrist angles for the same frames, extract ranges
    rng = np.random.RandomState(args.seed)
    n_wrist = 50000
    wrist_pitch, wrist_yaw = [], []
    for p_mm, y_mm, v_mm in [(pitch_r, yaw_r, v_wrist_r), (pitch_l, yaw_l, v_wrist_l)]:
        idx = np.sort(rng.choice(len(p_mm), size=min(n_wrist, len(p_mm)), replace=False))
        hand_idx = idx[v_mm[idx]]
        p = np.deg2rad(np.array(p_mm[hand_idx]))
        y = np.deg2rad(np.array(y_mm[hand_idx]))
        finite = np.isfinite(p) & np.isfinite(y)
        wrist_pitch.append(p[finite])
        wrist_yaw.append(y[finite])
    all_wrist_pitch = np.concatenate(wrist_pitch)
    all_wrist_yaw = np.concatenate(wrist_yaw)
    wrist_fe_range = float(all_wrist_pitch.max() - all_wrist_pitch.min())
    wrist_ru_range = float(all_wrist_yaw.max() - all_wrist_yaw.min())

    ego_ranges_22 = np.concatenate([ego_finger_ranges, [wrist_fe_range, wrist_ru_range]])

    # Finger-only totals for the summary text
    emg_finger_total = emg_finger_ranges.sum()
    ego_finger_total = ego_finger_ranges.sum()
    ego_total_22 = ego_ranges_22.sum()
    print(f"  EMG2Pose finger-only range: {emg_finger_total:.1f} rad", file=sys.stderr)
    print(f"  EgoEmg  finger-only range:  {ego_finger_total:.1f} rad", file=sys.stderr)
    print(f"  EgoEmg  22-joint range:     {ego_total_22:.1f} rad", file=sys.stderr)
    print(f"  Wrist F/E range: {wrist_fe_range:.2f} rad  R/U range: {wrist_ru_range:.2f} rad",
          file=sys.stderr)

    plot_joint_range(emg_ranges_22, ego_ranges_22, out_dir / "fig_joint_range_comparison.pdf")

    # ── Figure 2: Gesture t-SNE (EgoEmg only) ──
    print("Generating gesture t-SNE figure...", file=sys.stderr)
    ego_gc_sample = np.array(ego_gc_mm[ego_global_idx % len(ego_gc_mm)])
    ego_ga_sample = np.array(ego_ga_mm[ego_global_idx % len(ego_ga_mm)])
    plot_gesture_tsne(ego_ja_sample, ego_vm_sample, ego_gc_sample, ego_ga_sample,
                      out_dir / "fig_gesture_tsne.pdf")

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
