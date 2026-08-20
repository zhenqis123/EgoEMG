# Backward-compatible re-export of the classic plotly visualization helpers
# (formerly egoemg/visualization.py) and the EgoEMG-specific renderers.
#
# Imports are resolved lazily (PEP 562): pulling in `viz_utils` (which needs
# only numpy + cv2) must not eagerly import matplotlib/plotly/pyrender/smplx
# from the other submodules, so lightweight consumers such as
# `scripts/viz/visualize_dataset.py --help` work without the `viz` extra.
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from egoemg.visualization.classic import (
        get_plotly_animation_for_joint_angles,
        joint_angles_to_frames,
        plot_hand_mesh,
    )
    from egoemg.visualization.egoemg_vis import EgoEmgVisualizer
    from egoemg.visualization.mesh_renderer import ManoMeshRenderer

__all__ = [
    "EgoEmgVisualizer",
    "ManoMeshRenderer",
    "get_plotly_animation_for_joint_angles",
    "joint_angles_to_frames",
    "plot_hand_mesh",
]

_SUBMODULE_NAMES = {"classic", "viz_utils", "mesh_renderer", "egoemg_vis"}
_CLASSIC_EXPORTS = {
    "get_plotly_animation_for_joint_angles",
    "joint_angles_to_frames",
    "plot_hand_mesh",
}
_RENDERER_EXPORTS = {
    "ManoMeshRenderer": "mesh_renderer",
    "EgoEmgVisualizer": "egoemg_vis",
}


def __getattr__(name: str) -> Any:
    # Resolved attributes are cached into module globals so later accesses
    # bypass this hook. Submodules MUST be imported via importlib: the
    # `from package import submodule` form would recurse into __getattr__.
    if name in _SUBMODULE_NAMES:
        mod = importlib.import_module(f"egoemg.visualization.{name}")
        globals()[name] = mod
        return mod
    if name in _CLASSIC_EXPORTS or name not in _RENDERER_EXPORTS:
        classic = importlib.import_module("egoemg.visualization.classic")
        if name in _CLASSIC_EXPORTS or hasattr(classic, name):
            attr = getattr(classic, name)
            globals()[name] = attr
            return attr
    else:
        module_name = _RENDERER_EXPORTS[name]
        mod = importlib.import_module(f"egoemg.visualization.{module_name}")
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | _SUBMODULE_NAMES)
