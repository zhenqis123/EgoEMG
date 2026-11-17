"""Published session metadata and array access shared by all modalities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

SPLIT_IDS = {
    "train": 0,
    "user": 1,
    "gesture": 2,
    "both": 3,
    "recent_object": 4,
    "recent_user": 5,
    "recent_user_object": 6,
    "recent_test": 7,
}

VAL_SPLITS = ("user", "gesture", "both", "recent_object")

TEST_SPLITS = VAL_SPLITS


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(folder: Path, spec: dict[str, Any]) -> np.memmap:
    return np.memmap(
        str(folder / spec["path"]),
        dtype=spec["dtype"],
        mode="r",
        shape=tuple(spec["shape"]),
    )


def _valid_label_key(arrays: dict[str, Any]) -> str:
    if "joint_angles_valid" in arrays:
        return "joint_angles_valid"
    if "label_valid" in arrays:
        return "label_valid"
    raise ValueError("A session hand has no joint-angle validity field")
