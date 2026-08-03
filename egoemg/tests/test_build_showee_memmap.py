import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare" / "build_showee_memmap.py"
SPEC = importlib.util.spec_from_file_location("build_showee_memmap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_left_emg_permutation_direction() -> None:
    source = np.arange(16, dtype=np.float32).reshape(2, 8)
    actual = source[:, MODULE.LEFT_EMG_PERMUTATION]
    expected = np.asarray(
        [[1, 0, 7, 6, 5, 4, 3, 2], [9, 8, 15, 14, 13, 12, 11, 10]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_nearest_indices() -> None:
    source = np.asarray([0, 10, 20], dtype=np.int64)
    target = np.asarray([-2, 4, 6, 15, 22], dtype=np.int64)
    np.testing.assert_array_equal(
        MODULE._nearest_indices(source, target),
        np.asarray([0, 0, 1, 1, 2]),
    )


def test_emg_nanovolts_to_microvolts() -> None:
    source = np.asarray([-1000.0, 0.0, 2500.0], dtype=np.float32)
    actual = source * MODULE.EMG_NANOVOLTS_TO_MICROVOLTS
    np.testing.assert_allclose(actual, np.asarray([-1.0, 0.0, 2.5]))
