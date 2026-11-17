#!/usr/bin/env python3
"""Compute per-hand EMG normalization from session-v2 train rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def source_signature(dataset_dir, field, session_id_prefix):
    root = Path(dataset_dir).resolve()
    manifest = json.loads((root / "dataset.json").read_text())
    sources = {"manifest": manifest, "field": field, "prefix": session_id_prefix,
               "sessions": []}
    for entry in manifest["sessions"]:
        path = root / entry["path"]
        session = json.loads(path.read_text())
        if session_id_prefix is not None and not session["session_id"].startswith(session_id_prefix):
            continue
        files = {}
        for state in session["hands"].values():
            if state:
                for key in (field, "split_id", "emg_valid"):
                    if key in state["arrays"]:
                        p = path.parent / state["arrays"][key]["path"]
                        stat = p.stat()
                        files[str(p)] = [stat.st_size, stat.st_mtime_ns]
        sources["sessions"].append(dict(session=session, files=files))
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def compute(dataset_dir: Path, output: Path, field: str = "emg_filtered",
            session_id_prefix: str | None = None) -> dict:
    root = dataset_dir.resolve()
    manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if manifest.get("units", {}).get("emg") != "mV":
        raise ValueError("Expected EMG in mV")
    signature = source_signature(root, field, session_id_prefix)
    selected_sessions = []
    totals = {
        hand: dict(count=0, sum=np.zeros(8), squared=np.zeros(8))
        for hand in ("left", "right")
    }
    for entry in manifest["sessions"]:
        path = root / entry["path"]
        session = json.loads(path.read_text(encoding="utf-8"))
        if session_id_prefix is not None and not session["session_id"].startswith(session_id_prefix):
            continue
        selected_sessions.append(session["session_id"])
        for hand, state in session["hands"].items():
            if not state:
                continue
            arrays = state["arrays"]
            spec = arrays[field]
            if spec.get("unit") != "mV":
                raise ValueError(f"Unnormalized EMG: {path} {hand}")
            folder = path.parent
            n = state["rows"]
            emg = np.memmap(
                folder / spec["path"], dtype="float32", mode="r", shape=(n, 8)
            )
            split = np.memmap(
                folder / arrays["split_id"]["path"],
                dtype="int8",
                mode="r",
                shape=(n,),
            )
            valid = None
            if "emg_valid" in arrays:
                valid_spec = arrays["emg_valid"]
                valid = np.memmap(
                    folder / valid_spec["path"],
                    dtype=valid_spec["dtype"],
                    mode="r",
                    shape=(n,),
                )
            total = totals[hand]
            for start in range(0, n, 262_144):
                stop = min(n, start + 262_144)
                keep = np.asarray(split[start:stop]) == 0
                if valid is not None:
                    keep &= np.asarray(valid[start:stop], dtype=bool)
                if not keep.any():
                    continue
                chunk = np.asarray(emg[start:stop][keep], dtype=np.float64)
                if not np.isfinite(chunk).all():
                    raise ValueError(f"Nonfinite train EMG: {path} {hand} row {start}")
                total["count"] += len(chunk)
                total["sum"] += chunk.sum(axis=0)
                total["squared"] += np.square(chunk).sum(axis=0)
            del emg, split, valid
    stats = {}
    for hand, total in totals.items():
        count = total["count"]
        if not count:
            raise ValueError(f"No train EMG for {hand}")
        mean = total["sum"] / count
        std = np.sqrt(np.maximum(total["squared"] / count - mean**2, 0))
        if np.any(std <= 0):
            raise ValueError(f"Zero variance EMG channel for {hand}")
        stats[f"{field}_{hand}"] = dict(
            mean=mean.tolist(),
            std=std.tolist(),
            rows=count,
            split="train",
            unit="mV",
            valid_emg_only=True,
            session_id_prefix=session_id_prefix,
            session_ids=selected_sessions,
            source_signature=signature,
        )
    if source_signature(root, field, session_id_prefix) != signature:
        raise RuntimeError("Normalization source changed during computation")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    temporary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/EgoEMG_dataset"))
    parser.add_argument("--session-id-prefix", default=None)
    parser.add_argument("--reuse-if-current", action="store_true")
    parser.add_argument(
        "--field", choices=("emg_filtered", "emg_raw"), default="emg_filtered"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to the active protocol's stats path, or the historical v2 path",
    )
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "dataset.json").read_text())
    default_output = Path(
        manifest.get("split_policy", {}).get(
            "recent_train_norm_stats", "assets/egoemg_session_v2_emg_stats.json"
        )
    )
    if args.field == "emg_raw":
        default_output = default_output.with_name(
            default_output.stem.removesuffix("_stats")
            + "_raw_stats"
            + default_output.suffix
        )
    output = args.output or default_output
    result = None
    if args.reuse_if_current and output.exists():
        candidate = json.loads(output.read_text())
        signature = source_signature(args.dataset, args.field, args.session_id_prefix)
        if all(candidate.get(f"{args.field}_{side}", {}).get("source_signature") == signature
               for side in ("left", "right")):
            result = candidate
            print(f"Reusing verified normalization: {output}")
    if result is None:
        result = compute(args.dataset, output, field=args.field,
                         session_id_prefix=args.session_id_prefix)
    print(
        {
            key: {"rows": value["rows"], "std": value["std"]}
            for key, value in result.items()
        }
    )
