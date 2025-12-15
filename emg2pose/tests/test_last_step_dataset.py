from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from emg2pose.data import LastStepValidMultiSessionWindowDataset


def test_last_step_dataset_fixed_length_with_jitter_and_workers():
    hdf5_path = Path(__file__).parent / "assets" / "test_data.hdf5"
    ds = LastStepValidMultiSessionWindowDataset(
        hdf5_paths=[hdf5_path],
        window_length=50,
        stride=20,
        padding=(0, 0),
        jitter=True,
        allow_mask_recompute=True,  # test asset has no precomputed masks
        treat_interpolated_as_valid=False,
        # Asset joint angles may be all-zeros; only test fixed-length + worker-collate.
        require_last_step_valid=False,
        require_finite_last_step=True,
        max_open_sessions=2,
        show_progress=False,
    )

    assert len(ds) > 0
    sample0 = ds[0]
    assert sample0["emg"].shape[-1] == 50
    # Last-step dataset only returns supervision for the final timestamp.
    assert sample0["joint_angles"].shape[-1] == 1
    assert sample0["no_ik_failure"].shape[-1] == 1
    assert "interpolated_mask" not in sample0

    # Reproduce the worker-collate path; should not throw.
    try:
        loader = DataLoader(ds, batch_size=4, num_workers=1, shuffle=True)
        batch = next(iter(loader))
        assert isinstance(batch["emg"], torch.Tensor)
        assert batch["emg"].shape[-1] == 50
        assert batch["joint_angles"].shape[-1] == 1
        assert batch["no_ik_failure"].shape[-1] == 1
    except PermissionError:
        pytest.skip("Multiprocessing semaphores not permitted in this environment.")
