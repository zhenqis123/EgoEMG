"""Shared label conversion for pretraining corpora."""

import numpy as np


def pimforce_to_emg2pose_angles(
    pose: np.ndarray, *, pose_in_degrees: bool
) -> np.ndarray:
    """
    Convert Pimforce angle ordering to emg2pose joint ordering.

    PiMforce order: per finger [spread, flex1, flex2, flex3]
    emg2pose order: see egoemg/constants.py
    """
    pose = pose.copy()
    if pose_in_degrees:
        pose[0] -= 20.0
        pose[1] -= 60.0
    else:
        pose[0] -= np.deg2rad(20.0)
        pose[1] -= np.deg2rad(60.0)
    if pose_in_degrees:
        pose = np.deg2rad(pose)

    idx_map = np.array(
        [
            1,
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ],
        dtype=np.int64,
    )
    return pose[idx_map, ...]
