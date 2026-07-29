#!/usr/bin/env python
"""Compute unified per_dataset_norm_stats for the merged EgoEMG+ShowEE+Incre memmap.

The merged dataset spans three sources with different EMG scales (EgoEMG and
Incre share the FFT filter_paper pipeline; ShowEE is Wavelet-scaled to uV).
This produces row-weighted per-channel mean/std under a single
``egoemg_unified__filtered_paper[_left/_right]`` key set, so the merged dataset
can normalize every sample with one consistent stat block.

Row weighting uses each source's recorded sample count (`n_samples` for EgoEMG/
Incre, `num_sampled_rows` for ShowEE's sampled estimate — ShowEE's full row
count is used when available to avoid the sampling underestimate).

Usage:
    python scripts/data/compute_unified_norm_stats.py \
        --input assets/per_dataset_norm_stats_repro_filtered_paper_alias.json \
        --output assets/per_dataset_norm_stats_unified.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

# Full row counts per source (from the manifests) — used for row weighting.
ROW_COUNTS = {
    "egoemg": 66_161_725,
    "showee": 38_631_093,
    "egoemg_incre": 27_673_226,
}


def _n(entry: dict) -> int:
    for k in ("n_samples", "num_sampled_rows", "n_full"):
        if k in entry:
            return int(entry[k])
    return 0


def merge_channel_stats(entries: list[dict]) -> dict:
    """Row-weighted merge of per-channel mean/std across sources.

    Assumes EMG filtered signals are zero-mean within each source (the recorded
    means are ~1e-11).  The merged mean is the row-weighted average; the merged
    per-channel std is the row-weighted RMS of the per-channel stds (correct
    under the zero-mean, equal-variance-within-source assumption).

    Handles entries that only carry scalar mean/std (no per-channel breakdown)
    by falling back to scalar-only merging.
    """
    entries = [e for e in entries if e]
    if not entries:
        return {}
    weights = [_n(e) for e in entries]
    total = sum(weights)
    if total == 0:
        weights = [1] * len(entries)
        total = len(entries)

    merged_mean = sum(w * e["mean"] for w, e in zip(weights, entries)) / total
    merged_std = (sum(w * e["std"] ** 2 for w, e in zip(weights, entries)) / total) ** 0.5

    out = {
        "mean": merged_mean,
        "std": merged_std,
        "n_samples": total,
        "note": (
            "row-weighted merge of egoemg + showee + egoemg_incre "
            f"(weights={weights})"
        ),
    }
    # Per-channel breakdown only if every source provides it.
    chan_entries = [e for e in entries if "per_channel_mean" in e]
    if chan_entries and len(chan_entries) == len(entries):
        n_chan = len(chan_entries[0]["per_channel_mean"])
        out["per_channel_mean"] = [
            sum(w * e["per_channel_mean"][c] for w, e in zip(weights, entries)) / total
            for c in range(n_chan)
        ]
        out["per_channel_std"] = [
            (sum(w * e["per_channel_std"][c] ** 2 for w, e in zip(weights, entries))
             / total) ** 0.5
            for c in range(n_chan)
        ]
    return out


def _src(entry: dict) -> str:
    return entry.get("note", "").split(",")[0].split(" ")[0] or "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    stats = json.load(open(args.input))
    out = copy.deepcopy(stats)

    # Left hand: EgoEMG-left + ShowEE-left (Incre is right-only).
    left_entries = [
        stats.get("egoemg__filtered_paper_left"),
        stats.get("showee__filtered_paper_left"),
    ]
    # Right hand: EgoEMG-right + ShowEE-right + Incre-right.
    right_entries = [
        stats.get("egoemg__filtered_paper_right"),
        stats.get("showee__filtered_paper_right"),
        stats.get("egoemg_incre__filtered_paper_right"),
    ]
    # Combined (both hands pooled) for the dataset-wide key.
    both_entries = [
        stats.get("egoemg__filtered_paper"),
        stats.get("showee__filtered_paper"),
    ]

    # Patch missing row counts with authoritative ROW_COUNTS so weighting is
    # correct even when a source entry omits n_samples (e.g. showee__filtered_paper).
    order = ["egoemg", "showee", "egoemg_incre"]
    for entries, names in [
        (left_entries, ["egoemg", "showee"]),
        (right_entries, ["egoemg", "showee", "egoemg_incre"]),
        (both_entries, ["egoemg", "showee"]),
    ]:
        for e, name in zip(entries, names):
            if e and _n(e) == 0:
                e["n_samples"] = ROW_COUNTS[name]

    out["egoemg_unified__filtered_paper_left"] = merge_channel_stats(left_entries)
    out["egoemg_unified__filtered_paper_right"] = merge_channel_stats(right_entries)
    out["egoemg_unified__filtered_paper"] = merge_channel_stats(both_entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"Wrote unified norm stats -> {args.output}")
    for k in (
        "egoemg_unified__filtered_paper_left",
        "egoemg_unified__filtered_paper_right",
        "egoemg_unified__filtered_paper",
    ):
        e = out[k]
        pcs = [round(s, 3) for s in e["per_channel_std"]] if "per_channel_std" in e else None
        print(f"  {k}: mean={e['mean']:.3e} std={e['std']:.4f} "
              f"n={e['n_samples']:,} per_ch_std={pcs}")


if __name__ == "__main__":
    main()
