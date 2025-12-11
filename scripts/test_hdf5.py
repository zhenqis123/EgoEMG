import numpy as np
import h5py
def print_hdf5(name, obj):
    print(name)
file = h5py.File("data/emg2pose_data/2022-04-07-1649318400-8125c-cv-emg-pose-train@2-recording-1_left.hdf5")
with h5py.File("data/emg2pose_data/2022-04-07-1649318400-8125c-cv-emg-pose-train@2-recording-1_left.hdf5", "r") as f:
    data = f['emg2pose/timeseries']
    print(np.array(data["emg"]).shape)
    attrs = f['emg2pose'].attrs
    for key, value in attrs.items():
        print(key, ":", value)
