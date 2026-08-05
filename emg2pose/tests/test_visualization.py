# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import emg2pose.visualization as visualization
import plotly.graph_objects as go


def test_hand_mesh_plot():

    N_JOINT_ANGLES = 22
    N_EXPECTED_VERTS = 788

    joint_angles = np.zeros((N_JOINT_ANGLES,))
    joint_angles[5] = -5
    joint_angles[6] = -5

    fig = visualization.plot_hand_mesh(
        joint_angles, flip=True, opacity=1.0, color="lightpink"
    )

    assert isinstance(fig, go.Figure)
    assert fig.data[0].x.shape == (N_EXPECTED_VERTS,)
