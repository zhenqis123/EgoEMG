"""
Generate appendix materials for per-gesture modality complementarity analysis.

Reads per_gesture_analysis.csv and produces:
  - tables/per_gesture_family.tex   (family-level summary, sample-weighted)
  - tables/per_gesture_full.tex     (full 60-gesture breakdown)
  - figures/per_gesture_scatter.pdf (Vision vs Fusion scatter per gesture)

Usage:
    python scripts/paper/generate_per_gesture_appendix.py \
        --input-csv ./per_gesture_test_resnet/per_gesture_analysis.csv \
        --output-dir ./paper
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Gesture vocabulary: ID → (short_name, description) ───────────────────
GESTURE_NAMES = {
    0:  ("ASL1",                "ASL digit 1"),
    1:  ("ASL2",                "ASL digit 2"),
    2:  ("ASL3",                "ASL digit 3"),
    3:  ("ASL4",                "ASL digit 4"),
    4:  ("ASL5",                "ASL digit 5"),
    5:  ("ASL6",                "ASL digit 6"),
    6:  ("ASL7",                "ASL digit 7"),
    7:  ("ASL8",                "ASL digit 8"),
    8:  ("ASL9",                "ASL digit 9"),
    9:  ("Claw3",               "Three-finger claw pose (index + middle + ring flexed)"),
    10: ("Claw5",               "Five-finger claw pose (all fingers flexed)"),
    11: ("FreeAction",          "Free-form unconstrained hand action"),
    12: ("ILY",                 "I Love You hand sign (thumb, index, pinky extended)"),
    13: ("IndexBow",            "Index finger flexion/extension bowing"),
    14: ("IndexMiddleClaw",     "Index and middle fingers clawed, others extended"),
    15: ("JoystickCircle",      "Circular joystick manipulation on the palm with thumb"),
    16: ("JoystickSlide",       "Linear joystick sliding on the palm with thumb"),
    17: ("MiddleBow",           "Middle finger flexion/extension bowing"),
    18: ("Nine",                "Hand forming number 9 (four fingers curled, index up)"),
    19: ("PalmYaw",             "Palm facing up/down rotation about the yaw axis"),
    20: ("PinchMiddle",         "Thumb to middle fingertip pinch"),
    21: ("PinkyBow",            "Pinky finger flexion/extension bowing"),
    22: ("Rest",                "Relaxed hand in neutral resting posture"),
    23: ("RingAndThumb",        "Ring finger touching thumb tip"),
    24: ("RingBow",             "Ring finger flexion/extension bowing"),
    25: ("Rock",                "Rock on hand sign (index and pinky extended)"),
    26: ("Thumb",               "Thumb up"),
    27: ("nocontact_disperse_palm", "Fingers spread apart with palm open, no contact"),
    28: ("nocontact_free",       "Free-form finger motion without contact"),
    29: ("nocontact_grab",       "Simulated grasping motion without object contact"),
    30: ("Clap",                "Both hands clapping together symmetrically"),
    31: ("CrossHand",           "Both hands crossed fingers"),
    32: ("CrossStretch",        "Hands opposite fingers and stretched"),
    33: ("FingerPullLeft",      "Left hand pulls right-hand fingers"),
    34: ("FingerPullRight",     "Right hand pulls left-hand fingers"),
    35: ("FingerTipTouch",      "Matching fingertips of both hands touching"),
    36: ("FistBump",            "Two fists bumping together"),
    37: ("Gaming",              "Both hands holding a game controller"),
    38: ("HandClasp",           "Hands clasped together"),
    39: ("HandRub",             "Rubbing palms together"),
    40: ("IndexTapping",        "Both index fingers tapping in the air"),
    41: ("PalmRoll",            "Palms facing, rolling around each other"),
    42: ("PalmStack",           "One palm stacked on top of the other"),
    43: ("PinkyHook",           "Pinky fingers hooked together"),
    44: ("Prayer",              "Palms pressed together in prayer position"),
    45: ("Squeeze",             "Both hands squeezing an imaginary object"),
    46: ("SymOpen",             "Both hands opening symmetrically from closed to open"),
    47: ("SymSwing",            "Both hands swinging symmetrically side to side"),
    48: ("ThumbWrestle",        "Thumbs wrestling each other"),
    49: ("Typing",              "Both hands typing on a virtual keyboard"),
    50: ("raw",                 "Unconstrained bimanual hand movement"),
    51: ("Kiss",                "Fingertips of both hands touching (thumb and middle), then separating"),
    52: ("MiddleOppo",          "Middle and ring finger of one hand opposes the other hand"),
    53: ("Beijing",             "Two hands crossing to make a 'Bei' Chinese character"),
    54: ("Checky",              "Two hands thumb make 'Checky' sign"),
    55: ("PairClaw",            "Both hands clawing with asymmetric finger configurations"),
    56: ("PairOK",              "Both hands forming OK signs"),
    57: ("Picture",             "Hands forming a picture frame rectangle"),
    58: ("PinchWring",          "Two hand pinch and wring"),
    59: ("ThumbOppo",           "Thumbs of both hands in opposition at different orientations"),
}

# ── Gesture families ─────────────────────────────────────────────────────
# Matches vocabulary table in Appendix~\ref{app:gesture_vocabulary}
FAMILIES = {
    "Single-hand":           list(range(0, 30)),   # 0--29
    "Symmetric bimanual":    [30, 31, 32, 35, 36, 37, 38, 39, 40, 42, 44, 46, 47, 48, 49, 50, 51, 52],
    "Asymmetric bimanual":   [33, 34, 41, 43, 45, 53, 54, 55, 56, 57, 58, 59],
}

FAMILY_ORDER = list(FAMILIES.keys())


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def load_csv(path: str) -> list[dict]:
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_family_stats(rows: list[dict]) -> dict:
    """Compute per-family sample-weighted statistics."""
    # Collect per-gesture data
    gc_data: dict[int, dict] = {}
    for r in rows:
        gc = int(r["gesture_class"])
        if gc < 0:
            continue
        gc_data[gc] = {
            "vis": float(r["vision_mae_deg"]),
            "emg": float(r["emg_mae_deg"]),
            "fus": float(r["fusion_mae_deg"]),
            "n": int(r["n"]),
        }

    gc_family: dict[int, str] = {}
    for fname, gids in FAMILIES.items():
        for gid in gids:
            gc_family[gid] = fname

    stats = {}
    for family in FAMILY_ORDER:
        gids = [g for g in FAMILIES[family] if g in gc_data]
        if not gids:
            continue

        # Sample-weighted averages
        total_n = sum(gc_data[g]["n"] for g in gids)
        vis_w = sum(gc_data[g]["vis"] * gc_data[g]["n"] for g in gids) / total_n
        emg_w = sum(gc_data[g]["emg"] * gc_data[g]["n"] for g in gids) / total_n
        fus_w = sum(gc_data[g]["fus"] * gc_data[g]["n"] for g in gids) / total_n
        delta = vis_w - fus_w

        stats[family] = {
            "n_gestures": len(gids),
            "n_samples": total_n,
            "vis_avg": vis_w,
            "emg_avg": emg_w,
            "fus_avg": fus_w,
            "delta": delta,
        }
    return stats


def generate_family_table(stats: dict, output_dir: Path):
    """LaTeX family-summary table with sample-weighted averages."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-gesture-family modality comparison on \dataset{} (F-RN18+S "
        r"fusion). Values are sample-weighted averages across all test splits; "
        r"MAE is in degrees (lower is better). "
        r"$\Delta = \text{MAE}_\text{vision} - \text{MAE}_\text{fusion}$; "
        r"positive $\Delta$ indicates fusion improvement. "
        r"Gesture families follow the vocabulary in "
        r"Appendix~\ref{app:gesture_vocabulary}.}"
    )
    lines.append(r"\label{tab:per_gesture_family}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3.8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Gesture family & \#G & \#Samp. & Vis. & EMG & Fusion & $\Delta$ \\"
    )
    lines.append(r"\midrule")

    # Identify largest delta for bold
    max_delta = max(s["delta"] for s in stats.values())

    total_n_g = 0
    total_n_s = 0
    total_vis_w = 0.0
    total_emg_w = 0.0
    total_fus_w = 0.0

    for family in FAMILY_ORDER:
        s = stats[family]
        total_n_g += s["n_gestures"]
        total_n_s += s["n_samples"]
        total_vis_w += s["vis_avg"] * s["n_samples"]
        total_emg_w += s["emg_avg"] * s["n_samples"]
        total_fus_w += s["fus_avg"] * s["n_samples"]

        delta_str = f"{s['delta']:+.2f}"
        if abs(s["delta"] - max_delta) < 0.005:
            delta_str = rf"\mathbf{{{delta_str}}}"

        lines.append(
            f"{family} & {s['n_gestures']} & {s['n_samples']:,} & "
            f"{s['vis_avg']:.1f} & {s['emg_avg']:.1f} & {s['fus_avg']:.1f} & "
            f"${delta_str}$ \\\\"
        )

    # Overall
    total_vis_w /= total_n_s
    total_emg_w /= total_n_s
    total_fus_w /= total_n_s
    overall_delta = total_vis_w - total_fus_w

    lines.append(r"\midrule")
    lines.append(
        f"Overall & {total_n_g} & {total_n_s:,} & "
        f"{total_vis_w:.1f} & {total_emg_w:.1f} & {total_fus_w:.1f} & "
        f"${overall_delta:+.2f}$ \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    (output_dir / "tables" / "per_gesture_family.tex").write_text("\n".join(lines))
    print("  Wrote tables/per_gesture_family.tex")

    # Print preview
    print("\n  Family summary preview:")
    for family in FAMILY_ORDER:
        s = stats[family]
        print(
            f"  {family:28s}  G={s['n_gestures']:2d}  N={s['n_samples']:5d}  "
            f"V={s['vis_avg']:.1f}  E={s['emg_avg']:.1f}  F={s['fus_avg']:.1f}  "
            f"Δ={s['delta']:+.2f}"
        )
    print(
        f"  {'Overall':28s}  G={total_n_g:2d}  N={total_n_s:5d}  "
        f"V={total_vis_w:.1f}  E={total_emg_w:.1f}  F={total_fus_w:.1f}  "
        f"Δ={overall_delta:+.2f}"
    )


def generate_full_table(rows: list[dict], output_dir: Path):
    """LaTeX table: all gestures, grouped by family, sorted by delta desc."""
    gc_rows = {}
    for r in rows:
        gc = int(r["gesture_class"])
        if gc >= 0:
            gc_rows[gc] = r

    gc_family = {}
    for fname, gids in FAMILIES.items():
        for gid in gids:
            gc_family[gid] = fname

    family_idx = {f: i for i, f in enumerate(FAMILY_ORDER)}

    def sort_key(gc):
        fam = gc_family.get(gc, "ZZZ")
        fi = family_idx.get(fam, 99)
        r = gc_rows.get(gc)
        if r is None:
            return (fi, 0)
        delta = float(r["vision_mae_deg"]) - float(r["fusion_mae_deg"])
        return (fi, -delta)

    sorted_gc = sorted(gc_rows.keys(), key=sort_key)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Full per-gesture modality breakdown on \dataset{} "
        r"(F-RN18+S fusion). MAE is in degrees. "
        r"$\Delta = \text{MAE}_\text{vis} - \text{MAE}_\text{fus}$. "
        r"Gestures grouped by family, sorted by descending $\Delta$ "
        r"(largest fusion gain first) within each family. "
        r"Gesture descriptions are provided in "
        r"Appendix~\ref{app:gesture_vocabulary}.}"
    )
    lines.append(r"\label{tab:per_gesture_full}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.2pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.02}")
    lines.append(r"\begin{tabular}{rllrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"ID & Name & Family & $N$ & Vis. & EMG & Fus. & $\Delta$ \\"
    )
    lines.append(r"\midrule")

    last_family = None
    for gc in sorted_gc:
        r = gc_rows[gc]
        name, _desc = GESTURE_NAMES.get(gc, (f"G{gc}", ""))
        name = latex_escape(name)
        family = gc_family.get(gc, "---")
        n = int(r["n"])
        vis = float(r["vision_mae_deg"])
        emg = float(r["emg_mae_deg"])
        fus = float(r["fusion_mae_deg"])
        delta = vis - fus

        if last_family is not None and family != last_family:
            lines.append(r"\addlinespace[3pt]")
        last_family = family

        lines.append(
            f"{gc} & {name} & {family} & {n} & "
            f"{vis:.1f} & {emg:.1f} & {fus:.1f} & {delta:+.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    (output_dir / "tables" / "per_gesture_full.tex").write_text("\n".join(lines))
    print("  Wrote tables/per_gesture_full.tex")


def generate_scatter(rows: list[dict], output_dir: Path):
    """Per-gesture scatter: Vision MAE vs Fusion MAE, colored by family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gc_rows = {}
    for r in rows:
        gc = int(r["gesture_class"])
        if gc >= 0:
            gc_rows[gc] = r

    gc_family = {}
    for fname, gids in FAMILIES.items():
        for gid in gids:
            gc_family[gid] = fname

    family_colors_plot = {
        "Single-hand":           "#1f77b4",
        "Symmetric bimanual":    "#8c564b",
        "Asymmetric bimanual":   "#9467bd",
    }

    legend_order = FAMILY_ORDER  # use same order as table; \\& works in matplotlib

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, ax = plt.subplots(figsize=(5.0, 4.4))

    max_val = 0.0
    for _gc, r in gc_rows.items():
        max_val = max(max_val, float(r["vision_mae_deg"]), float(r["fusion_mae_deg"]))
    lim = math.ceil(max_val) + 1
    ax.set_xlim(2.5, lim)
    ax.set_ylim(2.5, lim)

    # Diagonal
    ax.plot([2.5, lim], [2.5, lim], "k--", linewidth=0.7, alpha=0.45, zorder=1)

    # Region labels
    ax.text(lim - 0.6, lim - 0.35, "Fusion worse",
            fontsize=7, color="gray", ha="right", va="top", style="italic")
    ax.text(3.4, 2.8, "Fusion better",
            fontsize=7, color="gray", ha="left", va="bottom", style="italic")

    # Plot by family
    for fname in legend_order:
        gids = FAMILIES.get(fname, [])
        xs, ys, sizes = [], [], []
        for gc in gids:
            if gc not in gc_rows:
                continue
            r = gc_rows[gc]
            xs.append(float(r["vision_mae_deg"]))
            ys.append(float(r["fusion_mae_deg"]))
            sizes.append(max(18, min(130, np.sqrt(int(r["n"])) * 4.5)))
        color = family_colors_plot.get(fname, "#999999")
        ax.scatter(
            xs, ys, s=sizes, c=color, edgecolors="white",
            linewidth=0.5, alpha=0.88, zorder=2, label=fname,
        )

    # Annotate top-4 improved and all gestures where fusion is worse
    deltas = []
    for gc, r in gc_rows.items():
        delta = float(r["vision_mae_deg"]) - float(r["fusion_mae_deg"])
        deltas.append((delta, gc))
    deltas.sort(key=lambda x: x[0], reverse=True)

    for delta, gc in deltas[:4]:
        r = gc_rows[gc]
        name, _desc = GESTURE_NAMES.get(gc, (f"G{gc}", ""))
        ax.annotate(
            name,
            (float(r["vision_mae_deg"]), float(r["fusion_mae_deg"])),
            textcoords="offset points",
            xytext=(7, -4),
            fontsize=6.2, color="#1a6b1a",
            arrowprops=dict(arrowstyle="->", color="#1a6b1a", lw=0.6),
        )

    for delta, gc in deltas:
        if delta >= 0:
            break
        r = gc_rows[gc]
        name, _desc = GESTURE_NAMES.get(gc, (f"G{gc}", ""))
        ax.annotate(
            name,
            (float(r["vision_mae_deg"]), float(r["fusion_mae_deg"])),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=6.2, color="#8b0000",
            arrowprops=dict(arrowstyle="->", color="#8b0000", lw=0.6),
        )

    ax.set_xlabel("Vision-only MAE (\N{DEGREE SIGN})")
    ax.set_ylabel("Fusion MAE (\N{DEGREE SIGN})")
    ax.set_aspect("equal")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(
        [by_label[k] for k in legend_order if k in by_label],
        [k for k in legend_order if k in by_label],
        loc="lower right", framealpha=0.92, edgecolor="gray",
        handletextpad=0.4, borderpad=0.3,
    )
    leg.set_zorder(10)

    fig.tight_layout(pad=0.6)
    fig_path = output_dir / "figures" / "per_gesture_scatter.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  Wrote figures/per_gesture_scatter.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="./paper")
    args = parser.parse_args()

    rows = load_csv(args.input_csv)
    print(f"Loaded {len(rows)} gesture rows from {args.input_csv}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = compute_family_stats(rows)
    generate_family_table(stats, out)
    generate_full_table(rows, out)
    generate_scatter(rows, out)

    print("\nDone. Add to appendix:")
    print("  \\input{tables/per_gesture_family}")
    print("  \\input{tables/per_gesture_full}")
    print("  \\includegraphics{figures/per_gesture_scatter.pdf}")


if __name__ == "__main__":
    main()
