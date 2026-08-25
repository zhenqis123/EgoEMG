"""Compute EMG signal statistics across datasets using sequential chunk reading."""

import numpy as np
import json
import os
from pathlib import Path


def load_memmap_field(manifest_path, field="emg"):
    with open(manifest_path) as f:
        manifest = json.load(f)
    info = manifest["fields"][field]
    base_dir = os.path.dirname(manifest_path)
    filepath = os.path.join(base_dir, info["filename"])
    arr = np.memmap(filepath, dtype=info["dtype"], mode="r", shape=tuple(info["shape"]))
    return arr


def compute_stats_sequential(arr, name, chunk_size=1_000_000, max_samples=5_000_000):
    """Compute stats using sequential chunks (Welford-like online algorithm)."""
    n = arr.shape[0]
    n_channels = arr.shape[1] if arr.ndim > 1 else 1

    print(f"\n{'='*60}", flush=True)
    print(f"Dataset: {name}", flush=True)
    print(f"  Total samples: {n:,}, Channels: {n_channels}", flush=True)

    # Read at most max_samples in sequential chunks
    n_read = min(n, max_samples)
    n_chunks = (n_read + chunk_size - 1) // chunk_size

    # Collect all samples for stats
    samples = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_read)
        chunk = arr[start:end]
        samples.append(chunk)
        if (i + 1) % 10 == 0 or i == n_chunks - 1:
            print(f"  Read chunk {i+1}/{n_chunks} ({end:,}/{n_read:,} samples)", flush=True)

    data = np.concatenate(samples, axis=0)
    print(f"  Sampled: {data.shape[0]:,}", flush=True)

    # Global stats
    print(f"\n  Global statistics:", flush=True)
    print(f"    mean:    {data.mean():.6f}", flush=True)
    print(f"    std:     {data.std():.4f}", flush=True)
    print(f"    min:     {data.min():.4f}", flush=True)
    print(f"    max:     {data.max():.4f}", flush=True)
    for p in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]:
        print(f"    p{p:05.1f}:   {np.percentile(data, p):.4f}", flush=True)

    # Per-channel stats
    if arr.ndim > 1:
        print(f"\n  Per-channel statistics:", flush=True)
        print(f"  {'Ch':>3s}  {'mean':>10s}  {'std':>8s}  {'min':>8s}  {'max':>8s}  {'p01':>8s}  {'p99':>8s}", flush=True)
        ch_stds = []
        for c in range(n_channels):
            ch = data[:, c]
            ch_std = ch.std()
            ch_stds.append(ch_std)
            print(f"  {c:3d}  {ch.mean():10.6f}  {ch_std:8.4f}  {ch.min():8.4f}  {ch.max():8.4f}  {np.percentile(ch, 1):8.4f}  {np.percentile(ch, 99):8.4f}", flush=True)

        print(f"\n  Channel std range: [{min(ch_stds):.4f}, {max(ch_stds):.4f}]", flush=True)
        print(f"  Channel std ratio (max/min): {max(ch_stds)/min(ch_stds):.2f}x", flush=True)

    return {
        "name": name,
        "total_samples": n,
        "channels": n_channels,
        "global_mean": float(data.mean()),
        "global_std": float(data.std()),
        "global_min": float(data.min()),
        "global_max": float(data.max()),
        "percentiles": {p: float(np.percentile(data, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
    }


def main():
    results = []

    # EMG corpus root: override via EMG_CORPUS_ROOT env var; defaults to a
    # sibling ``data/emg_corpus`` directory relative to the repo root.
    emg_corpus = Path(os.environ.get(
        "EMG_CORPUS_ROOT",
        str(Path(__file__).resolve().parents[2] / "data" / "emg_corpus"),
    ))

    datasets = [
        ("emg2pose_v3", emg_corpus / "emg2pose_memmap" / "manifest.json", "emg"),
        ("pimforce_v3", emg_corpus / "pimforce_v3_memmap" / "manifest.json", "emg"),
        ("ninapro_DB1", emg_corpus / "Ninapro_relabeled_memmap" / "DB1" / "manifest.json", "emg"),
        ("ninapro_DB2", emg_corpus / "Ninapro_relabeled_memmap" / "DB2" / "manifest.json", "emg"),
        ("ninapro_DB5", emg_corpus / "Ninapro_relabeled_memmap" / "DB5" / "manifest.json", "emg"),
    ]

    # emg2qwerty has separate left/right
    qwerty_manifest = emg_corpus / "emg2qwerty_v3_memmap" / "manifest.json"
    with open(qwerty_manifest) as f:
        m = json.load(f)
    for side in ["left", "right"]:
        key = f"emg_{side}"
        if key in m["fields"]:
            datasets.append((f"emg2qwerty_{side}", qwerty_manifest, key))

    for name, manifest, field in datasets:
        arr = load_memmap_field(manifest, field)
        results.append(compute_stats_sequential(arr, name))

    # Summary comparison
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Dataset':<25s} {'Ch':>3s} {'Mean':>10s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'P1':>8s} {'P99':>8s}", flush=True)
    for r in results:
        p = r["percentiles"]
        print(f"  {r['name']:<25s} {r['channels']:>3d} {r['global_mean']:>10.6f} {r['global_std']:>8.4f} {r['global_min']:>8.4f} {r['global_max']:>8.4f} {p[1]:>8.4f} {p[99]:>8.4f}", flush=True)

    # Save to CSV
    import pandas as pd
    rows = []
    for r in results:
        row = {"dataset": r["name"], "channels": r["channels"],
               "mean": r["global_mean"], "std": r["global_std"],
               "min": r["global_min"], "max": r["global_max"]}
        for p, v in r["percentiles"].items():
            row[f"p{p}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = "emg_stats_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
