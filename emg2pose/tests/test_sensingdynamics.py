import math

import torch

from emg2pose.models.featurizers.sensingdynamics import (
    SMU,
    SensingDynamicsConvBlock,
    SensingDynamicsFeaturizer,
    SensingDynamicsMLP,
    butterworth_lowpass,
)
from emg2pose.models.modules.sensingdynamics import SensingDynamicsModule


def test_smu_matches_relu_limit() -> None:
    activation = SMU(alpha=0.0, mu=100.0, learnable=False)
    values = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
    torch.testing.assert_close(
        activation(values), torch.relu(values), atol=1e-5, rtol=1e-5
    )


def test_lowpass_attenuates_high_frequency() -> None:
    sample_rate = 2000.0
    time = torch.arange(4000) / sample_rate
    low = torch.sin(2.0 * math.pi * 10.0 * time)
    high = torch.sin(2.0 * math.pi * 200.0 * time)
    signal = (low + high)[None, None]
    filtered = butterworth_lowpass(signal, sample_rate, 20.0, order=4)

    low_projection = (filtered * low).mean().abs()
    high_projection = (filtered * high).mean().abs()
    assert low_projection > 100.0 * high_projection


def _make_model() -> SensingDynamicsModule:
    featurizer = SensingDynamicsFeaturizer(
        conv_blocks=[
            SensingDynamicsConvBlock(2, 8, (3, 5), stride=(2, 2)),
            SensingDynamicsConvBlock(8, 16, (3, 5), stride=(2, 2)),
            SensingDynamicsConvBlock(16, 32, (3, 5), stride=(2, 1)),
        ],
        out_channels=32,
    )
    decoder = SensingDynamicsMLP(32, 22, hidden_channels=64)
    return SensingDynamicsModule(featurizer, decoder, out_channels=22)


def test_sensingdynamics_forward_shapes() -> None:
    model = _make_model()
    batch = {
        "emg": torch.randn(2, 8, 512),
        "joint_angles": torch.randn(2, 22, 512),
        "label_valid_mask": torch.ones(2, 512, dtype=torch.bool),
    }
    prediction, target, mask = model(batch)
    assert prediction.shape == target.shape == (2, 22, 512)
    assert mask.shape == (2, 512)


def test_sensingdynamics_does_not_require_initial_pose() -> None:
    model = _make_model().eval()
    emg = torch.randn(1, 8, 512)
    base = {
        "emg": emg,
        "joint_angles": torch.zeros(1, 22, 512),
        "label_valid_mask": torch.ones(1, 512, dtype=torch.bool),
    }
    shifted = {**base, "joint_angles": torch.full((1, 22, 512), 7.0)}
    with torch.no_grad():
        first = model(base)[0]
        second = model(shifted)[0]
    torch.testing.assert_close(first, second)
