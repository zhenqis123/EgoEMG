"""Local MANO wrapper used by EgoEMG vision supervision.

This intentionally does not import WiLoR. It mirrors the small behavior needed
from WiLoR's MANO wrapper: append fingertip vertices to the 16 MANO joints and
return the 21 joints in OpenPose hand order.
"""

from __future__ import annotations

import pickle
from typing import Optional

import smplx
import torch
from smplx.lbs import vertices2joints
from smplx.utils import MANOOutput, to_tensor
from smplx.vertex_ids import vertex_ids


MANO_TO_OPENPOSE = [
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
]


class OpenPoseMANO(smplx.MANOLayer):
    """MANOLayer with WiLoR/OpenPose-compatible 21-joint output."""

    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if joint_regressor_extra is not None:
            with open(joint_regressor_extra, "rb") as f:
                regressor = pickle.load(f, encoding="latin1")
            self.register_buffer(
                "joint_regressor_extra",
                torch.tensor(regressor, dtype=torch.float32),
            )
        self.register_buffer(
            "extra_joints_idxs",
            to_tensor(list(vertex_ids["mano"].values()), dtype=torch.long),
        )
        self.register_buffer(
            "joint_map",
            torch.tensor(MANO_TO_OPENPOSE, dtype=torch.long),
        )

    def forward(self, *args, **kwargs) -> MANOOutput:
        mano_output = super().forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, "joint_regressor_extra"):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output
