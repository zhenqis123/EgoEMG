# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import numpy as np
import pytest
from emg2pose.data import Emg2PoseSessionData


metadata_keys = [
    "filename",
    "end",
    "num_channels",
    "sample_rate",
    "session",
    "side",
    "stage",
    "start",
    "user",
]
metadata_types = [
    str,
    float,
    np.int64,
    float,
    str,
    str,
    str,
    float,
    str,
]
timeseries_keys = ["time", "emg", "joint_angles"]
timeseries_shapes = [(), (16,), (20,)]


@pytest.fixture
def test_hdf5_path() -> str:
    test_hdf5_path = Path(__file__).parent / "assets" / "test_data.hdf5"
    return str(test_hdf5_path)


def test_Emg2PoseSessionData(test_hdf5_path):
    session = Emg2PoseSessionData(test_hdf5_path)

    assert isinstance(session.metadata, dict)
    for key, expected_type in zip(metadata_keys, metadata_types):
        assert key in session.metadata
        assert isinstance(
            session.metadata[key], expected_type
        ), f"{key} has incorrect type"

    T = len(session.timeseries)
    assert isinstance(T, int)
    assert T > 0

    for key, expected_shape in zip(timeseries_keys, timeseries_shapes):
        assert isinstance(session.timeseries[key], np.ndarray)
        assert session.timeseries[key].shape == (T,) + expected_shape  # (T, N)
