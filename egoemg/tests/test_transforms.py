import numpy as np
import torch

from egoemg.transforms import (
    FromSpectrogram,
    RandomGain,
    RandomSpectrogramBandTimeMask,
    ToSpectrogram,
)
from egoemg.transforms_batch import BatchAugmentation


def test_random_gain_per_channel_in_range() -> None:
    np.random.seed(0)
    emg = np.ones((128, 8), dtype=np.float32)
    aug = RandomGain(
        min_gain=0.7,
        max_gain=1.4,
        mask_prob=1.0,
        per_channel=True,
        channel_dim=-1,
    )
    out = aug(emg)
    assert out.shape == emg.shape
    gains = out[0]
    assert np.allclose(out, gains[None, :])
    assert np.all(gains >= 0.7)
    assert np.all(gains <= 1.4)


def test_spectrogram_round_trip_dict_keeps_pose() -> None:
    torch.manual_seed(0)
    emg = torch.randn(256, 8)
    joint_angles = torch.randn(256, 20)
    sample = {"emg": emg.clone(), "joint_angles": joint_angles.clone()}

    to_spec = ToSpectrogram(n_fft=128, hop_length=32, center=True)
    from_spec = FromSpectrogram(n_fft=128, hop_length=32, center=True)

    spec_sample = to_spec(sample)
    assert "_emg_stft_length" in spec_sample
    assert torch.is_complex(spec_sample["emg"])
    assert torch.equal(spec_sample["joint_angles"], joint_angles)

    out_sample = from_spec(spec_sample)
    assert "_emg_stft_length" not in out_sample
    assert out_sample["emg"].shape == emg.shape
    assert torch.allclose(out_sample["emg"], emg, atol=1e-5, rtol=1e-5)
    assert torch.equal(out_sample["joint_angles"], joint_angles)


def test_spectrogram_block_mask_zeroes_some_bins() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    c, f, t = 4, 129, 122
    spec = torch.ones((c, f, t), dtype=torch.complex64)
    mask = RandomSpectrogramBandTimeMask(
        sample_rate=2000.0,
        n_fft=256,
        bands_hz=((125.0, 250.0),),
        band_weights=None,
        mask_prob=1.0,
        num_masks=1,
        min_time_mask_size=5,
        max_time_mask_size=5,
        per_channel=False,
        mask_value=0.0,
    )

    masked = mask(spec)
    assert masked.shape == spec.shape

    zeros = masked == 0
    assert int(zeros.sum().item()) > 0

    k_low = int(round(125.0 * 256 / 2000.0))
    k_high = int(round(250.0 * 256 / 2000.0))
    assert zeros[:, :k_low, :].sum().item() == 0
    assert zeros[:, k_high + 1 :, :].sum().item() == 0


def test_batch_time_mask_broadcasts_over_channels() -> None:
    torch.manual_seed(0)
    aug = BatchAugmentation({
        "time_mask": {
            "num_masks": 1,
            "min_mask_size": 5,
            "max_mask_size": 5,
            "per_channel": False,
        },
    })
    x = torch.ones(2, 8, 32)
    out = aug._time_mask(x)
    assert out.shape == x.shape
    expected = (out[:, :1] == 0).expand_as(out[:, 1:])
    assert torch.equal(expected, out[:, 1:] == 0)
