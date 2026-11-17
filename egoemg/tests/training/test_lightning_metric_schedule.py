from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from egoemg.training.prediction import _metric_contributes_to_loss
from egoemg.metrics import LandmarkDistances, get_default_metrics
from egoemg.train import make_lightning_module


def test_train_metric_schedule_keeps_weighted_losses() -> None:
    metrics = get_default_metrics()
    weights = {"mae": 1.0, "landmark/fingertip": 0.01}
    selected = {
        type(metric).__name__
        for metric in metrics
        if _metric_contributes_to_loss(metric, weights)
    }
    assert selected == {"AngleMAE", "LandmarkDistances"}


def test_train_metric_schedule_respects_other_loss_weights() -> None:
    metrics = get_default_metrics()
    selected = {
        type(metric).__name__
        for metric in metrics
        if _metric_contributes_to_loss(
            metric, {"mae_per_finger/thumb": 0.5, "derivatives/vel": 0.2}
        )
    }
    assert selected == {"PerFingerAngleMAE", "AngularDerivatives"}


@pytest.mark.parametrize(
    "experiment",
    ["emgformer_s"],
)
def test_egoemg_fingertip_loss_switch(experiment: str) -> None:
    config_dir = Path(__file__).resolve().parents[3] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        base = f"experiment=emgformer/{experiment}"
        disabled = compose(config_name="base", overrides=[base])
        enabled = compose(
            config_name="base",
            overrides=[base, "enable_fingertip_loss=true"],
        )

    off_module = make_lightning_module(disabled)
    assert not any(isinstance(m, LandmarkDistances) for m in off_module.regression_metrics)
    assert off_module.loss_weights.get("landmark/fingertip", 0.0) == 0.0
    assert off_module.loss_weights.get("fingertip_distance", 0.0) == 0.0

    on_module = make_lightning_module(enabled)
    assert any(isinstance(m, LandmarkDistances) for m in on_module.regression_metrics)
    fingertip_weight = sum(
        on_module.loss_weights.get(key, 0.0)
        for key in ("landmark/fingertip", "fingertip_distance")
    )
    assert fingertip_weight == pytest.approx(0.01)
