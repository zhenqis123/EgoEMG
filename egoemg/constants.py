# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass


EMG_SAMPLE_RATE = 2000
NUM_JOINTS = 20
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
PD_GROUPS = ["proximal", "mid", "distal"]


@dataclass
class Joint:
    name: str
    index: int
    groups: list[str]


# AA: abduction, adduction
# FE: flexion, extension
JOINTS: list[Joint] = [
    Joint("THUMB_CMC_FE", 0, ["thumb", "proximal"]),
    Joint("THUMB_CMC_AA", 1, ["thumb", "proximal"]),
    Joint("THUMB_MCP_FE", 2, ["thumb", "mid"]),
    Joint("THUMB_IP_FE", 3, ["thumb", "distal"]),
    Joint("INDEX_MCP_AA", 4, ["index", "proximal"]),
    Joint("INDEX_MCP_FE", 5, ["index", "proximal"]),
    Joint("INDEX_PIP_FE", 6, ["index", "mid"]),
    Joint("INDEX_DIP_FE", 7, ["index", "distal"]),
    Joint("MIDDLE_MCP_AA", 8, ["middle", "proximal"]),
    Joint("MIDDLE_MCP_FE", 9, ["middle", "proximal"]),
    Joint("MIDDLE_PIP_FE", 10, ["middle", "mid"]),
    Joint("MIDDLE_DIP_FE", 11, ["middle", "distal"]),
    Joint("RING_MCP_AA", 12, ["ring", "proximal"]),
    Joint("RING_MCP_FE", 13, ["ring", "proximal"]),
    Joint("RING_PIP_FE", 14, ["ring", "mid"]),
    Joint("RING_DIP_FE", 15, ["ring", "distal"]),
    Joint("PINKY_MCP_AA", 16, ["pinky", "proximal"]),
    Joint("PINKY_MCP_FE", 17, ["pinky", "proximal"]),
    Joint("PINKY_PIP_FE", 18, ["pinky", "mid"]),
    Joint("PINKY_DIP_FE", 19, ["pinky", "distal"]),
]


@dataclass
class Landmark:
    name: str
    index: int
    groups: list[str]


LANDMARKS: list[Landmark] = [
    Landmark("THUMB_FINGERTIP", 0, ["thumb", "fingertip"]),
    Landmark("INDEX_FINGER_FINGERTIP", 1, ["index", "fingertip"]),
    Landmark("MIDDLE_FINGER_FINGERTIP", 2, ["middle", "fingertip"]),
    Landmark("RING_FINGER_FINGERTIP", 3, ["ring", "fingertip"]),
    Landmark("PINKY_FINGER_FINGERTIP", 4, ["pinky", "fingertip"]),
    Landmark("WRIST_JOINT", 5, ["wrist"]),
    Landmark("THUMB_INTERMEDIATE_FRAME", 6, ["thumb"]),
    Landmark("THUMB_DISTAL_FRAME", 7, ["thumb"]),
    Landmark("INDEX_PROXIMAL_FRAME", 8, ["index"]),
    Landmark("INDEX_INTERMEDIATE_FRAME", 9, ["index"]),
    Landmark("INDEX_DISTAL_FRAME", 10, ["index"]),
    Landmark("MIDDLE_PROXIMAL_FRAME", 11, ["middle"]),
    Landmark("MIDDLE_INTERMEDIATE_FRAME", 12, ["middle"]),
    Landmark("MIDDLE_DISTAL_FRAME", 13, ["middle"]),
    Landmark("RING_PROXIMAL_FRAME", 14, ["ring"]),
    Landmark("RING_INTERMEDIATE_FRAME", 15, ["ring"]),
    Landmark("RING_DISTAL_FRAME", 16, ["ring"]),
    Landmark("PINKY_PROXIMAL_FRAME", 17, ["pinky"]),
    Landmark("PINKY_INTERMEDIATE_FRAME", 18, ["pinky"]),
    Landmark("PINKY_DISTAL_FRAME", 19, ["pinky"]),
    Landmark("PALM_CENTER", 20, ["palm"]),
]

# The following landmarks do not move because the wrist doesn't move. We therefore
# mask them in landmark metrics.
NO_MOVEMENT_LANDMARKS = [
    "INDEX_PROXIMAL_FRAME",
    "MIDDLE_PROXIMAL_FRAME",
    "RING_PROXIMAL_FRAME",
    "PINKY_PROXIMAL_FRAME",
]
