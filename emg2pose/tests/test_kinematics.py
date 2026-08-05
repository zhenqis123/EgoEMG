# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from emg2pose.kinematics import forward_kinematics


def test_basic_default_forward_kinematics():
    joint_angles = torch.randn((2, 20, 4))  # batch, 20 joints, 4 time steps
    lp = forward_kinematics(joint_angles)
    assert lp.shape == (2, 4, 21, 3)  # batch, 4 time steps, 21 joints, 3d
