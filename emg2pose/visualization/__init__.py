# Backward-compatible re-export of the classic plotly visualization helpers
# (formerly emg2pose/visualization.py).  Kept here so both
#   from emg2pose.visualization import plot_hand_mesh
# and the EgoEMG-specific EgoEmgVisualizer resolve from one package.
from emg2pose.visualization.classic import *  # noqa: F401,F403
from emg2pose.visualization.classic import (
    get_plotly_animation_for_joint_angles,
    joint_angles_to_frames,
    plot_hand_mesh,
)
from emg2pose.visualization.mesh_renderer import ManoMeshRenderer
from emg2pose.visualization.egoemg_vis import EgoEmgVisualizer
