from __future__ import annotations

import pytest
import torch

from egoemg.metrics import AnglularDerivatives, _masked_mae


def test_angular_derivatives_supports_single_timestep():
    metric = AnglularDerivatives()
    pred = torch.zeros(2, 20, 1)
    target = torch.zeros(2, 20, 1)
    mask = torch.ones(2, 1, dtype=torch.bool)
    out = metric(pred, target, mask, stage="test")
    assert (
        "test_derivatives/vel" in out
        and "test_derivatives/acc" in out
        and "test_derivatives/jerk" in out
    )


def test_angular_derivatives_supports_two_timesteps():
    metric = AnglularDerivatives()
    pred = torch.zeros(2, 20, 2)
    target = torch.zeros(2, 20, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    out = metric(pred, target, mask, stage="test")
    assert (
        "test_derivatives/vel" in out
        and "test_derivatives/acc" in out
        and "test_derivatives/jerk" in out
    )


@pytest.mark.parametrize("valid_fraction", [0.0, 0.25, 1.0])
def test_masked_mae_matches_indexed_l1_and_gradient(valid_fraction: float) -> None:
    generator = torch.Generator().manual_seed(123)
    pred = torch.randn(3, 4, 19, generator=generator)
    target = torch.randn(3, 4, 19, generator=generator)
    mask = torch.rand(3, 4, 19, generator=generator) < valid_fraction
    target[~mask] = float("nan")

    actual_pred = pred.clone().requires_grad_()
    actual = _masked_mae(actual_pred, target, mask)
    actual.backward()

    expected_pred = pred.clone().requires_grad_()
    expected = (
        torch.nn.functional.l1_loss(expected_pred[mask], target[mask])
        if mask.any()
        else expected_pred.sum() * 0.0
    )
    expected.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_pred.grad, expected_pred.grad)
