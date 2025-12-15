import torch

from emg2pose.metrics import AnglularDerivatives


def test_angular_derivatives_supports_single_timestep():
    metric = AnglularDerivatives()
    pred = torch.zeros(2, 20, 1)
    target = torch.zeros(2, 20, 1)
    mask = torch.ones(2, 1, dtype=torch.bool)
    out = metric(pred, target, mask, stage="test")
    assert "test_vel" in out and "test_acc" in out and "test_jerk" in out


def test_angular_derivatives_supports_two_timesteps():
    metric = AnglularDerivatives()
    pred = torch.zeros(2, 20, 2)
    target = torch.zeros(2, 20, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    out = metric(pred, target, mask, stage="test")
    assert "test_vel" in out and "test_acc" in out and "test_jerk" in out
