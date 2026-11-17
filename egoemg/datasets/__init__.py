"""Lazy dataset exports; optional corpus dependencies load on demand."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from egoemg.datasets.corpora.emg2pose import Emg2PoseDataset
    from egoemg.datasets.corpora.ninapro import (
        NinaproMemmapDataset,
        NinaproDB1MemmapDataset,
        NinaproDB2MemmapDataset,
        NinaproDB5MemmapDataset,
    )
    from egoemg.datasets.corpora.pimforce import PimforceMemmapDataset
    from egoemg.datasets.pretrain_wrapper import PretrainWrapperDataset

_EXPORTS = {
    "Emg2PoseDataset": "corpora.emg2pose",
    "NinaproMemmapDataset": "corpora.ninapro",
    "NinaproDB1MemmapDataset": "corpora.ninapro",
    "NinaproDB2MemmapDataset": "corpora.ninapro",
    "NinaproDB5MemmapDataset": "corpora.ninapro",
    "PimforceMemmapDataset": "corpora.pimforce",
    "PretrainWrapperDataset": "pretrain_wrapper",
}

__all__ = [
    "Emg2PoseDataset",
    "NinaproMemmapDataset",
    "NinaproDB1MemmapDataset",
    "NinaproDB2MemmapDataset",
    "NinaproDB5MemmapDataset",
    "PimforceMemmapDataset",
    "PretrainWrapperDataset",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
