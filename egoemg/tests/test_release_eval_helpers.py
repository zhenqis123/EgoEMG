"""Release-critical eval helpers: center-frame grid math + output paths.

Covers the pure logic of the unified center-frame evaluator (grid
construction, split filtering, window-fit exclusion) and the results.csv
output-path fallback, which had no test coverage before the release audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from egoemg.center_frame import REF_WL, collect_val_centers
from egoemg.test_analysis import _results_output_path


def _write_manifest(memmap_dir: Path, n_rows: int, fields: dict) -> None:
    manifest = {
        "total_rows": n_rows,
        "fields": fields,
        "version": "v2",
    }
    (memmap_dir / "manifest.json").write_text(json.dumps(manifest))


def _make_synthetic_grid_memmap(tmp_path: Path) -> Path:
    """Minimal memmap with only episode_index + frame_split_id (grid math)."""
    n_rows = 4 * REF_WL + 500  # ~3 full REF_WL windows + tail
    mm = tmp_path / "grid_memmap"
    mm.mkdir(parents=True, exist_ok=True)
    episode = np.memmap(mm / "episode_index.dat", dtype=np.int64, mode="w+", shape=(n_rows,))
    split = np.memmap(mm / "frame_split_id.dat", dtype=np.int8, mode="w+", shape=(n_rows,))
    episode[:] = 0  # single episode spanning all rows
    # Split pattern: row 0 = train(0), rows 1..n-2 = val(1), last row = train(0)
    split[:] = 1
    split[0] = 0
    split[-1] = 0
    split.flush()
    _write_manifest(
        mm,
        n_rows,
        {
            "episode_index": {"dtype": "int64", "shape": [n_rows]},
            "frame_split_id": {"dtype": "int8", "shape": [n_rows]},
        },
    )
    return mm


def test_collect_val_centers_grid_math() -> None:
    """Centers are REF_WL apart, restricted to val splits and in-episode windows."""
    mm = _make_synthetic_grid_memmap(Path(__file__).parent / "tmp_grid")
    try:
        centers = collect_val_centers(mm)
        # Single episode -> one entry; window starts at 0, step REF_WL.
        assert set(centers.keys()) == {0}
        ep_centers = [c for c, _ in centers[0]]
        # n_windows = floor((n_rows - REF_WL) / REF_WL) + 1
        n_rows = 4 * REF_WL + 500
        expected_windows = (n_rows - REF_WL) // REF_WL + 1
        assert len(ep_centers) == expected_windows
        # Spacing is exactly REF_WL, and centers are window midpoints.
        assert all(
            b - a == REF_WL for a, b in zip(ep_centers, ep_centers[1:])
        )
        # All centers must sit on val rows (split == 1).
        split = np.memmap(
            mm / "frame_split_id.dat", dtype=np.int8, mode="r", shape=(n_rows,)
        )
        assert all(split[c] == 1 for c in ep_centers)
    finally:
        import shutil

        shutil.rmtree(mm, ignore_errors=True)


def test_collect_val_centers_window_fit() -> None:
    """required_window_length drops windows that would exceed the episode."""
    mm = _make_synthetic_grid_memmap(Path(__file__).parent / "tmp_grid_fit")
    try:
        n_rows = 4 * REF_WL + 500
        # A window longer than the episode tail: last window start must be
        # such that start + required_window_length <= n_rows.
        required = REF_WL + 1000
        centers = collect_val_centers(mm, required_window_length=required)
        ep_centers = [c for c, _ in centers[0]]
        starts = [c - required // 2 for c in ep_centers]
        assert all(s + required <= n_rows for s in starts)
        assert len(ep_centers) < (n_rows - REF_WL) // REF_WL + 1  # some dropped
    finally:
        import shutil

        shutil.rmtree(mm, ignore_errors=True)


def test_results_output_path_falls_back_to_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a Hydra context, _results_output_path lands in the cwd."""
    import os

    monkeypatch.setenv("PWD", "/tmp/whatever")
    # Force HydraConfig.get to raise (no active Hydra run).
    import sys

    class _NoHydra:
        @staticmethod
        def get():
            raise RuntimeError("no hydra context")

    monkeypatch.setitem(sys.modules, "hydra.core.hydra_config", None)
    monkeypatch.setattr(
        "hydra.core.hydra_config.HydraConfig", _NoHydra, raising=False
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "egoemg.test_analysis.HydraConfig",
            _NoHydra,
            raising=False,
        )
        out = _results_output_path()
        assert out.endswith("results.csv")
        assert os.path.isabs(out)
