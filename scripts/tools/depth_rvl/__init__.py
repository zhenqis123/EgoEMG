"""Lossless RVL as an HDF5 filter; /depth remains uint16 [frames, height, width]."""

import os
from pathlib import Path

FILTER_ID = 40517
VERSION = 1
ROOT = Path(__file__).resolve().parents[3]


def register_rvl(required=True):
    """Load the native plugin in h5py's HDF5 instance, not a second HDF5 DLL."""
    import h5py

    if h5py.h5z.filter_avail(FILTER_ID):
        return True
    candidates = []
    if os.environ.get("COLLECTOR_RVL_PLUGIN_DIR"):
        candidates.append(Path(os.environ["COLLECTOR_RVL_PLUGIN_DIR"]))
    candidates.extend(
        [
            Path(__file__).parent / "native/lib",
            ROOT / "my_project/build_release_fresh/Release",
        ]
    )
    for directory in candidates:
        if directory.is_dir():
            h5py.h5pl.append(os.fsencode(directory))
            if h5py.h5z.filter_avail(FILTER_ID):
                return True
    if required:
        raise RuntimeError(
            "RVL plugin missing. Build RvlHdf5Filter or set COLLECTOR_RVL_PLUGIN_DIR"
        )
    return False


def dataset_options(height, width):
    register_rvl()
    pixels = int(height) * int(width)
    if not 0 < pixels <= 0xFFFFFFFF:
        raise ValueError("Invalid depth dimensions")
    return dict(
        chunks=(1, height, width),
        compression=FILTER_ID,
        compression_opts=(VERSION, pixels),
        fletcher32=True,
    )


def open_depth(path, mode="r", **kwargs):
    import h5py

    register_rvl(required=False)
    # Optional for capture-only installations; required by RVL+Zstd deliveries.
    try:
        import hdf5plugin  # noqa: F401 - registers filters in h5py's HDF5
    except ImportError:
        pass
    return h5py.File(path, mode, **kwargs)
