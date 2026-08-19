# Backward-compatible re-export of the classic plotly visualization helpers
# (formerly egoemg/visualization.py).  Kept here so both
#   from egoemg.visualization import plot_hand_mesh
# and the EgoEMG-specific EgoEmgVisualizer resolve from one package.
from egoemg.visualization.classic import *  # noqa: F401,F403
from egoemg.visualization.classic import (
    get_plotly_animation_for_joint_angles,
    joint_angles_to_frames,
    plot_hand_mesh,
)
from egoemg.visualization.mesh_renderer import ManoMeshRenderer
from egoemg.visualization.egoemg_vis import EgoEmgVisualizer

__all__ = [
    "EgoEmgVisualizer",
    "ManoMeshRenderer",
    "get_plotly_animation_for_joint_angles",
    "joint_angles_to_frames",
    "plot_hand_mesh",
]
