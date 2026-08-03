# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import h5py
import numpy as np


def create_a_sample_hdf5_file():

    T = 100
    timeseries_dtype = np.dtype(
        [("time", "<f8"), ("joint_angles", "<f8", (20,)), ("emg", "<f4", (16,))]
    )
    main_timeseries = np.array(np.empty((T,), dtype=timeseries_dtype))

    main_timeseries["time"] = np.arange(T, dtype="<f8")
    main_timeseries["emg"] = np.random.randn(T, 16)
    main_timeseries["joint_angles"] = np.random.randn(T, 20)

    metadata = {
        "filename": "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-9_left",
        "session": "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-9",
        "stage": "IndividualFingerPointingSnap_both",
        "user": "d387095792",
        "side": "right",
        "sample_rate": 2000.0,
        "num_channels": 16,
        "start": 1670313313.1833334,
        "end": 1670313374.1,
    }

    with h5py.File("../assets/test_data.hdf5", "w") as f:
        emg2pose = f.create_group("emg2pose")
        emg2pose["timeseries"] = main_timeseries
        emg2pose.attrs.update(metadata)


if __name__ == "__main__":
    create_a_sample_hdf5_file()
