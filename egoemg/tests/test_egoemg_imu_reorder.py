"""Tests for the EgoEMG imu channel-order fix helper.

The helper lives in ``scripts/prepare/fix_egoemg_imu_channel_order.py`` (not
in the package), so it is loaded via importlib.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "prepare"
    / "fix_egoemg_imu_channel_order.py"
)

spec = importlib.util.spec_from_file_location("fix_egoemg_imu_channel_order", _SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
reorder = mod.reorder_imu_channels


def test_reorder_maps_gyro_first_to_acc_first():
    rows = np.array(
        [
            [0.0, 0.1, 0.2, 9.3, -1.0, 2.0],  # gyro-first: [gyro, acc]
            [0.0, -0.3, 0.4, 9.4, -1.1, 2.1],
        ],
        dtype=np.float32,
    )
    fixed = reorder(rows)
    np.testing.assert_allclose(
        fixed,
        [
            [9.3, -1.0, 2.0, 0.0, 0.1, 0.2],
            [9.4, -1.1, 2.1, 0.0, -0.3, 0.4],
        ],
    )


def test_reorder_is_involutive():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(64, 6)).astype(np.float32)
    np.testing.assert_array_equal(reorder(reorder(rows)), rows)


def test_reorder_rejects_wrong_width():
    with np.testing.assert_raises(ValueError):
        reorder(np.zeros((4, 5), dtype=np.float32))


def test_reorder_moves_gravity_side():
    rng = np.random.default_rng(1)
    gyro = rng.normal(0, 0.2, (1000, 3))
    acc = rng.normal(0, 0.1, (1000, 3))
    acc[:, 2] += 9.3
    rows = np.concatenate([gyro, acc], axis=1).astype(np.float32)
    fixed = reorder(rows)
    acc_mag = np.linalg.norm(fixed[:, :3], axis=1)
    assert np.percentile(acc_mag, 50) > 8.0
