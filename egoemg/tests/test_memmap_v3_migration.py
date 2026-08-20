"""Tests for the v2 -> v3 unified-memmap schema migration.

Builds a tiny synthetic v2 memmap, runs the migration, and asserts every
property of the v3 redesign: renames, drops, validity collapse, the Incre
valid-bit fix, and the manifest rewrite.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/data/migrate_unified_memmap_v3.py"
spec = importlib.util.spec_from_file_location("migrate_v3", _SCRIPT)
migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate)

N = 64
RENAMES = migrate.RENAMES
DROPS = migrate.DROPS


def _build_v2(root: Path) -> None:
    root.mkdir(parents=True)
    fields: dict[str, dict] = {}
    rng = np.random.default_rng(0)

    def put(name: str, arr: np.ndarray) -> None:
        arr.tofile(root / f"{name}.dat")
        fields[name] = {
            "filename": f"{name}.dat",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }

    put("emg_left_raw", rng.normal(size=(N, 8)).astype(np.float32))
    put("emg_right_raw", rng.normal(size=(N, 8)).astype(np.float32))
    put("emg_right_filtered", rng.normal(size=(N, 8)).astype(np.float32))  # orphan, dropped
    for old in RENAMES:
        width = 6 if old.startswith("imu") else 1
        put(old, rng.normal(size=(N, width)).astype(np.float32))
    # validity: all-or-nothing (21,) rows, Incre rows True-but-zero keypoints
    valid = np.zeros((N, 21), dtype=bool)
    valid[: N // 2] = True  # EgoEMG rows valid; Incre rows invalid
    valid[N - 4 :] = True  # Incre rows with TRUE valid + zero keypoints (the bug)
    put("mocap_left_valid", valid)
    put("mocap_right_valid", valid.copy())
    for name in DROPS:
        put(name, np.zeros(N, dtype=np.int64 if "index" in name or name == "timestamp" else np.bool_))
    put("dataset_source_id", np.array([0] * (N - 8) + [2] * 8, dtype=np.int8))

    manifest = {
        "format_version": "egoemg_v2_memmap",
        "total_rows": N,
        "num_episodes": 2,
        "fields": fields,
        "episode_fields": {},
        "dataset_sources": {"0": "egoemg", "1": "showee", "2": "egoemg_incre"},
        "source_policies": {"egoemg_incre": "vision/mocap unavailable (zero-filled, stale=True, valid=False)"},
        "imu_semantics": {"imu": "left band", "imu_right": "right band"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _run(root: Path, apply: bool) -> None:
    cmd = [sys.executable, str(_SCRIPT), "--memmap-dir", str(root)]
    if apply:
        cmd.append("--apply")
    subprocess.run(cmd, check=True, capture_output=True)


def test_migration_dry_run_changes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "mm"
    _build_v2(root)
    _run(root, apply=False)
    m = json.loads((root / "manifest.json").read_text())
    assert m["format_version"] == "egoemg_v2_memmap"
    assert (root / "imu.dat").exists()
    assert (root / "task_index.dat").exists()


def test_migration_v3(tmp_path: Path) -> None:
    root = tmp_path / "mm"
    _build_v2(root)
    _run(root, apply=True)
    m = json.loads((root / "manifest.json").read_text())

    assert m["format_version"] == "egoemg_v3_memmap"
    # renames applied on disk and in the manifest
    for old, new in RENAMES.items():
        assert old not in m["fields"] and new in m["fields"]
        assert (root / f"{new}.dat").exists() and not (root / f"{old}.dat").exists()
    # drops removed
    for name in DROPS:
        assert name not in m["fields"] and not (root / f"{name}.dat").exists()
    # validity KEEPS the per-keypoint shape (ShowEE partial validity is real)
    for name in ("mocap_left_valid", "mocap_right_valid"):
        assert m["fields"][name]["shape"] == [N, 21]
    # Incre rows: valid forced False (the 4 trailing rows were the bug)
    src = np.fromfile(root / "dataset_source_id.dat", dtype=np.int8)
    v = np.fromfile(root / "mocap_left_valid.dat", dtype=bool).reshape(N, 21)
    assert not v[src == 2].any()
    assert v[: N // 2].all()
    # semantics + migration record
    assert "field_semantics" in m and "emg" in m["field_semantics"]
    assert m["schema_migration"]["to"] == "egoemg_v3_memmap"
    assert m["imu_semantics"]["imu_band_left"] == "left band"
    assert "v3" in m["source_policies"]["egoemg_incre"]

    # idempotence
    _run(root, apply=True)
    assert json.loads((root / "manifest.json").read_text())["format_version"] == "egoemg_v3_memmap"
