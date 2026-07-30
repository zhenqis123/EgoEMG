from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


FS = 2000.0
LOW_CUT = 20.0
LOW_TRANSITION = 5.0
HIGH_CUT = 850.0
HIGH_TRANSITION = 50.0
NOTCH_CONFIGS = (
    {"center": 50.0, "stop_half_width": 1.5, "transition_half_width": 1.5},
    {"center": 100.0, "stop_half_width": 1.5, "transition_half_width": 1.5},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter EMG into new parquet columns using FFT-domain notch + wide bandpass.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episodes", default="", help="Optional comma-separated episode ids")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())


def write_table_streaming(path: Path, row_group_tables: list[pa.Table]) -> None:
    metadata = pq.read_metadata(path)
    compression = metadata.row_group(0).column(0).compression if metadata.num_row_groups > 0 else "UNCOMPRESSED"
    codec = "NONE" if str(compression).upper() == "UNCOMPRESSED" else str(compression).lower()
    tmp = path.with_suffix(path.suffix + ".tmp")

    writer = pq.ParquetWriter(
        tmp,
        row_group_tables[0].schema,
        compression=codec,
        use_dictionary=False,
        write_statistics=True,
    )
    try:
        for table in row_group_tables:
            writer.write_table(table, row_group_size=table.num_rows)
    finally:
        writer.close()

    backup_once(path)
    tmp.replace(path)


def set_or_add_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx >= 0:
        return table.set_column(idx, name, values)
    return table.append_column(name, values)


def _column_to_numpy_2d(col: pa.ChunkedArray) -> np.ndarray:
    col = col.combine_chunks()
    flat = col.flatten().to_numpy(zero_copy_only=False)
    inner = len(col[0].as_py())
    return flat.reshape(-1, inner).astype(np.float32)


def to_fixed_list_arr_2d(arr_2d: np.ndarray, inner_size: int) -> pa.ListArray:
    n = arr_2d.shape[0]
    flat = pa.array(arr_2d.reshape(-1).astype(np.float32), type=pa.float32())
    offsets = pa.array(np.arange(0, (n + 1) * inner_size, inner_size, dtype=np.int32), type=pa.int32())
    return pa.ListArray.from_arrays(offsets, flat)


def smoothstep_cosine(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def build_emg_frequency_mask(n_samples: int, fs: float = FS) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = np.ones_like(freqs, dtype=np.float64)

    # Wide conservative bandpass with soft roll-offs.
    hp0 = max(0.0, LOW_CUT - LOW_TRANSITION)
    hp1 = LOW_CUT
    if hp1 > hp0:
        below = freqs <= hp0
        trans = (freqs > hp0) & (freqs < hp1)
        mask[below] = 0.0
        mask[trans] *= smoothstep_cosine((freqs[trans] - hp0) / (hp1 - hp0))

    lp0 = HIGH_CUT
    lp1 = min(fs * 0.5, HIGH_CUT + HIGH_TRANSITION)
    if lp1 > lp0:
        above = freqs >= lp1
        trans = (freqs > lp0) & (freqs < lp1)
        mask[above] = 0.0
        mask[trans] *= (1.0 - smoothstep_cosine((freqs[trans] - lp0) / (lp1 - lp0)))

    # Narrow notches to remove mains + 2nd harmonic while preserving nearby content.
    for cfg in NOTCH_CONFIGS:
        center = cfg["center"]
        stop_hw = cfg["stop_half_width"]
        trans_hw = cfg["transition_half_width"]
        stop_lo = center - stop_hw
        stop_hi = center + stop_hw
        trans_lo = stop_lo - trans_hw
        trans_hi = stop_hi + trans_hw

        hard = (freqs >= stop_lo) & (freqs <= stop_hi)
        left = (freqs > trans_lo) & (freqs < stop_lo)
        right = (freqs > stop_hi) & (freqs < trans_hi)
        mask[hard] = 0.0
        if np.any(left):
            mask[left] *= (1.0 - smoothstep_cosine((freqs[left] - trans_lo) / (stop_lo - trans_lo)))
        if np.any(right):
            mask[right] *= smoothstep_cosine((freqs[right] - stop_hi) / (trans_hi - stop_hi))

    return mask


def filter_emg_fft(x: np.ndarray, fs: float = FS) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=True)
    x = x.astype(np.float64, copy=False)
    mean = np.mean(x, axis=0, keepdims=True)
    x0 = x - mean
    mask = build_emg_frequency_mask(x0.shape[0], fs=fs)[:, None]
    spec = np.fft.rfft(x0, axis=0)
    spec *= mask
    y = np.fft.irfft(spec, n=x0.shape[0], axis=0)
    return y.astype(np.float32)


def detect_powerline_report(dataset_root: Path, episode_rows: list[dict]) -> dict[str, float]:
    sample_eps = [int(r["episode_id"]) for r in episode_rows[: min(10, len(episode_rows))]]
    peaks: dict[int, list[float]] = {50: [], 60: [], 100: [], 120: []}
    for ep in sample_eps:
        row = next(r for r in episode_rows if int(r["episode_id"]) == ep)
        pf = pq.ParquetFile(dataset_root / row["file_id"])
        t = pf.read_row_groups([0], columns=["observation.emg.left"], use_threads=True)
        x = _column_to_numpy_2d(t["observation.emg.left"])[:32768]
        for ch in range(x.shape[1]):
            sig = x[:, ch].astype(np.float64) - float(np.mean(x[:, ch]))
            spec = np.abs(np.fft.rfft(sig)) ** 2
            freqs = np.fft.rfftfreq(sig.shape[0], d=1.0 / FS)
            for target in peaks:
                band = (freqs >= target - 1.0) & (freqs <= target + 1.0)
                peaks[target].append(float(np.max(spec[band])) if np.any(band) else 0.0)
    return {str(k): float(np.median(v)) if v else 0.0 for k, v in peaks.items()}


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    episode_rows = pq.read_table(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    if args.episodes.strip():
        selected = {int(part.strip()) for part in args.episodes.split(",") if part.strip()}
        episode_rows = [row for row in episode_rows if int(row["episode_id"]) in selected]

    powerline_report = detect_powerline_report(dataset_root, episode_rows)
    print("powerline PSD median peaks:", powerline_report)

    summary: dict[str, dict[str, float]] = {}
    progress = tqdm(episode_rows, desc="Filtering EMG", unit="episode")
    for row in progress:
        episode_id = int(row["episode_id"])
        parquet_path = dataset_root / str(row["file_id"])
        pf = pq.ParquetFile(parquet_path)
        emg_table = pq.read_table(parquet_path, columns=["observation.emg.left", "observation.emg.right"])
        left = _column_to_numpy_2d(emg_table["observation.emg.left"])
        right = _column_to_numpy_2d(emg_table["observation.emg.right"])

        left_filtered = filter_emg_fft(left)
        right_filtered = filter_emg_fft(right)

        summary[str(episode_id)] = {
            "left_std_in_mv": float(np.std(left)),
            "left_std_out_mv": float(np.std(left_filtered)),
            "right_std_in_mv": float(np.std(right)),
            "right_std_out_mv": float(np.std(right_filtered)),
        }
        print(
            f"episode {episode_id:06d} "
            f"left_std {summary[str(episode_id)]['left_std_in_mv']:.3f}->{summary[str(episode_id)]['left_std_out_mv']:.3f} mV "
            f"right_std {summary[str(episode_id)]['right_std_in_mv']:.3f}->{summary[str(episode_id)]['right_std_out_mv']:.3f} mV"
        )
        progress.set_postfix({"episode": f"{episode_id:06d}"})

        if args.dry_run:
            continue

        row_group_tables: list[pa.Table] = []
        offset = 0
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_groups([rg_idx], use_threads=True)
            n = rg.num_rows
            rg = set_or_add_column(
                rg,
                "observation.emg.left_filtered",
                to_fixed_list_arr_2d(left_filtered[offset:offset + n], 8),
            )
            rg = set_or_add_column(
                rg,
                "observation.emg.right_filtered",
                to_fixed_list_arr_2d(right_filtered[offset:offset + n], 8),
            )
            row_group_tables.append(rg)
            offset += n
        del pf
        write_table_streaming(parquet_path, row_group_tables)

    if not args.dry_run:
        info_path = dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.setdefault("features", {})
        features["observation.emg.left_filtered"] = {"dtype": "float32", "shape": [8], "unit": "mV"}
        features["observation.emg.right_filtered"] = {"dtype": "float32", "shape": [8], "unit": "mV"}
        info.setdefault("emg_filter", {})
        info["emg_filter"] = {
            "fs_hz": FS,
            "powerline_hz_verified": 50,
            "powerline_psd_median_peaks": powerline_report,
            "pipeline": [
                "subtract per-channel mean",
                "narrow notch around 50 Hz",
                "narrow notch around 100 Hz",
                "wide bandpass 20-850 Hz with soft roll-off to 900 Hz",
                "no normalization",
            ],
            "implementation": "fft_frequency_mask",
        }
        backup_once(info_path)
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary_path = dataset_root / "meta" / "emg_filter_summary.json"
        if summary_path.exists():
            backup_once(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
