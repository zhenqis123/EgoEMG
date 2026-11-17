"""Model exports loaded only when requested."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from egoemg.models.modules.base import BaseModule
    from egoemg.models.modules.emgformer import Emg2PoseFormer
    from egoemg.models.modules.emgformer_pretrain import EmgformerPretrain
    from egoemg.models.modules.pose import (
        PoseModule,
        StatePoseModule,
        VEMG2PoseWithInitialState,
    )
    from egoemg.models.modules.sensingdynamics import SensingDynamicsModule

_EXPORTS = {
    "BaseModule": "egoemg.models.modules.base",
    "Emg2PoseFormer": "egoemg.models.modules.emgformer",
    "EmgformerPretrain": "egoemg.models.modules.emgformer_pretrain",
    "PoseModule": "egoemg.models.modules.pose",
    "StatePoseModule": "egoemg.models.modules.pose",
    "VEMG2PoseWithInitialState": "egoemg.models.modules.pose",
    "SensingDynamicsModule": "egoemg.models.modules.sensingdynamics",
}

__all__ = [
    "BaseModule",
    "Emg2PoseFormer",
    "EmgformerPretrain",
    "PoseModule",
    "StatePoseModule",
    "VEMG2PoseWithInitialState",
    "SensingDynamicsModule",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
