# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import pywt
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class RandomGain:
    """Randomly scale EMG amplitude (gain augmentation).

    Applies a multiplicative gain to the EMG signal to simulate changes in
    electrode impedance and amplifier gain. Gain is sampled log-uniformly to
    better match multiplicative variability.
    """

    min_gain: float = 0.8
    max_gain: float = 1.25
    mask_prob: float = 0.5
    per_channel: bool = True
    channel_dim: int = -1

    def __post_init__(self) -> None:
        if self.min_gain <= 0 or self.max_gain <= 0:
            raise ValueError("min_gain and max_gain must be > 0.")
        if self.max_gain < self.min_gain:
            raise ValueError("max_gain must be >= min_gain.")
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1].")

    def __call__(
        self, data: np.ndarray | dict[str, Any]
    ) -> np.ndarray | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data

        if self.mask_prob == 0.0 or np.random.rand() > self.mask_prob:
            return data

        log_min = float(np.log(self.min_gain))
        log_max = float(np.log(self.max_gain))
        if self.per_channel:
            n_channels = data.shape[self.channel_dim]
            gains = np.exp(np.random.uniform(log_min, log_max, size=(n_channels,)))
            shape = [1] * data.ndim
            shape[self.channel_dim] = n_channels
            gains = gains.reshape(shape)
        else:
            gains = float(np.exp(np.random.uniform(log_min, log_max)))

        return (data * gains).astype(data.dtype, copy=False)


@dataclass
class ToSpectrogram:
    """Convert time-domain EMG to a complex STFT spectrogram (Torch)."""

    n_fft: int = 256
    hop_length: int = 64
    win_length: int | None = None
    window: str = "hann"
    center: bool = True
    onesided: bool = True
    time_dim: int = 0
    channel_dim: int = -1
    length_key: str = "_emg_stft_length"

    def __post_init__(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("n_fft must be > 0.")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be > 0.")
        if self.win_length is not None and self.win_length <= 0:
            raise ValueError("win_length must be > 0 when provided.")

    def __call__(
        self, data: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data[self.length_key] = int(emg.shape[self.time_dim])
            data["emg"] = self(emg)
            return data

        emg = data
        if emg.ndim != 2:
            raise ValueError(f"Expected 2D EMG tensor, got shape {tuple(emg.shape)}")

        time_dim = self.time_dim if self.time_dim >= 0 else emg.ndim + self.time_dim
        channel_dim = (
            self.channel_dim if self.channel_dim >= 0 else emg.ndim + self.channel_dim
        )
        if {time_dim, channel_dim} != {0, 1}:
            emg = emg.movedim((time_dim, channel_dim), (0, 1))

        # Now (T, C) -> (C, T) for batched STFT.
        emg_ct = emg.transpose(0, 1).contiguous()
        win_length = self.win_length if self.win_length is not None else self.n_fft
        # Ensure window is created on the same device as the input tensor
        window = (
            torch.hann_window(win_length, device=emg.device, dtype=emg.dtype)
            if self.window == "hann"
            else None
        )
        if window is None:
            raise ValueError(f"Unsupported window type: {self.window}")

        spec = torch.stft(
            emg_ct,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=win_length,
            window=window,
            center=self.center,
            onesided=self.onesided,
            return_complex=True,
        )
        return spec

import torchaudio.transforms as T

@dataclass
class RandomSpectrogramBlockMask:
    """Randomly mask rectangular blocks in a complex spectrogram (SpecAugment)."""

    mask_prob: float = 0.7
    num_masks: int = 2
    max_freq_mask_size: int = 64
    max_time_mask_size: int = 40
    min_freq_mask_size: int = 0
    min_time_mask_size: int = 0
    per_channel: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1].")
        if self.num_masks < 0:
            raise ValueError("num_masks must be >= 0.")
        if self.max_freq_mask_size < 0 or self.max_time_mask_size < 0:
            raise ValueError("max_*_mask_size must be >= 0.")
        if self.min_freq_mask_size < 0 or self.min_time_mask_size < 0:
            raise ValueError("min_*_mask_size must be >= 0.")
        if self.max_freq_mask_size < self.min_freq_mask_size:
            raise ValueError("max_freq_mask_size must be >= min_freq_mask_size.")
        if self.max_time_mask_size < self.min_time_mask_size:
            raise ValueError("max_time_mask_size must be >= min_time_mask_size.")

    def __call__(
        self, data: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data

        spec = data
        if self.num_masks == 0 or self.mask_prob == 0.0:
            return spec
        if spec.ndim not in (2, 3):
            raise ValueError(
                f"Expected spectrogram with 2 or 3 dims, got {tuple(spec.shape)}"
            )

        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        c, f, t = spec.shape
        if f == 0 or t == 0:
            return data

        out = spec.clone()
        # Ensure fill_value is created on the same device as the input tensor
        fill_value = out.new_tensor(self.mask_value)

        def _apply_mask(ch: int | None) -> None:
            for _ in range(self.num_masks):
                if np.random.rand() > self.mask_prob:
                    continue
                freq_size = (
                    self.max_freq_mask_size
                    if self.max_freq_mask_size == self.min_freq_mask_size
                    else np.random.randint(
                        self.min_freq_mask_size, self.max_freq_mask_size + 1
                    )
                )
                time_size = (
                    self.max_time_mask_size
                    if self.max_time_mask_size == self.min_time_mask_size
                    else np.random.randint(
                        self.min_time_mask_size, self.max_time_mask_size + 1
                    )
                )
                freq_size = int(min(freq_size, f))
                time_size = int(min(time_size, t))
                if freq_size == 0 or time_size == 0:
                    continue

                f0 = int(np.random.randint(0, f - freq_size + 1))
                t0 = int(np.random.randint(0, t - time_size + 1))

                if ch is None:
                    out[:, f0 : f0 + freq_size, t0 : t0 + time_size] = fill_value
                else:
                    out[ch, f0 : f0 + freq_size, t0 : t0 + time_size] = fill_value

        if self.per_channel:
            for ch in range(c):
                _apply_mask(ch)
        else:
            _apply_mask(None)

        if squeezed:
            return out.squeeze(0)
        return out


@dataclass
class RandomSpectrogramBandTimeMask:
    """Mask full frequency bands over random time spans in a complex spectrogram.

    Each mask selects one band from ``bands_hz`` (in Hz), converts it to STFT bin
    indices using ``sample_rate`` and ``n_fft``, then masks the full band across a
    contiguous time interval. This is designed to simulate band-limited dropouts
    or aggressive filtering artifacts.
    """

    sample_rate: float = 2000.0
    n_fft: int = 256
    bands_hz: Sequence[Sequence[float]] = (
        (31.25, 62.5),
        (62.5, 125.0),
        (125.0, 250.0),
        (250.0, 375.0),
        (375.0, 687.5),
        (687.5, 1000.0),
    )
    band_weights: Sequence[float] | None = None
    mask_prob: float = 0.7
    num_masks: int = 2
    max_time_mask_size: int = 30
    min_time_mask_size: int = 0
    per_channel: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0.")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be > 0.")
        if not self.bands_hz:
            raise ValueError("bands_hz must be non-empty.")
        if self.band_weights is not None and len(self.band_weights) != len(self.bands_hz):
            raise ValueError("band_weights must match bands_hz length.")
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1].")
        if self.num_masks < 0:
            raise ValueError("num_masks must be >= 0.")
        if self.max_time_mask_size < 0 or self.min_time_mask_size < 0:
            raise ValueError("time mask sizes must be >= 0.")
        if self.max_time_mask_size < self.min_time_mask_size:
            raise ValueError("max_time_mask_size must be >= min_time_mask_size.")

        parsed: list[tuple[float, float]] = []
        for band in self.bands_hz:
            if len(band) != 2:
                raise ValueError("Each band in bands_hz must have 2 elements.")
            low, high = float(band[0]), float(band[1])
            if low < 0 or high < 0:
                raise ValueError("Band frequencies must be >= 0.")
            if high <= low:
                raise ValueError("Each band must satisfy high > low.")
            parsed.append((low, high))
        self._bands_hz = parsed

        if self.band_weights is None:
            self._band_weights = None
        else:
            weights = np.asarray(list(self.band_weights), dtype=float)
            if np.any(weights < 0) or not np.isfinite(weights).all():
                raise ValueError("band_weights must be finite and >= 0.")
            if weights.sum() == 0:
                raise ValueError("band_weights must sum to > 0.")
            self._band_weights = (weights / weights.sum()).tolist()

    def __call__(
        self, data: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data

        spec = data
        if self.num_masks == 0 or self.mask_prob == 0.0:
            return spec
        if spec.ndim not in (2, 3):
            raise ValueError(
                f"Expected spectrogram with 2 or 3 dims, got {tuple(spec.shape)}"
            )

        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        c, f_bins, t_bins = spec.shape
        if f_bins == 0 or t_bins == 0:
            return data

        out = spec.clone()
        # Ensure fill_value is created on the same device as the input tensor
        fill_value = out.new_tensor(self.mask_value)

        def _hz_to_bin(freq_hz: float) -> int:
            idx = int(np.round(freq_hz * float(self.n_fft) / float(self.sample_rate)))
            return int(np.clip(idx, 0, f_bins - 1))

        def _pick_band_bins() -> tuple[int, int]:
            band_idx = int(
                np.random.choice(len(self._bands_hz), p=self._band_weights)
            )
            low_hz, high_hz = self._bands_hz[band_idx]
            lo = _hz_to_bin(low_hz)
            hi = _hz_to_bin(high_hz)
            if hi < lo:
                lo, hi = hi, lo
            return lo, hi

        def _pick_time_slice() -> slice | None:
            if self.max_time_mask_size == 0:
                return None
            if self.max_time_mask_size == self.min_time_mask_size:
                time_size = self.max_time_mask_size
            else:
                time_size = int(
                    np.random.randint(
                        self.min_time_mask_size, self.max_time_mask_size + 1
                    )
                )
            time_size = int(min(time_size, t_bins))
            if time_size == 0:
                return None
            t0 = int(np.random.randint(0, t_bins - time_size + 1))
            return slice(t0, t0 + time_size)

        def _apply_masks(ch: int | None) -> None:
            for _ in range(self.num_masks):
                if np.random.rand() > self.mask_prob:
                    continue
                time_slice = _pick_time_slice()
                if time_slice is None:
                    continue
                f0, f1 = _pick_band_bins()
                if ch is None:
                    out[:, f0 : f1 + 1, time_slice] = fill_value
                else:
                    out[ch, f0 : f1 + 1, time_slice] = fill_value

        if self.per_channel:
            for ch in range(c):
                _apply_masks(ch)
        else:
            _apply_masks(None)

        if squeezed:
            return out.squeeze(0)
        return out


@dataclass
class FromSpectrogram:
    """Convert complex STFT spectrogram back to time-domain EMG (Torch)."""

    n_fft: int = 256
    hop_length: int = 64
    win_length: int | None = None
    window: str = "hann"
    center: bool = True
    onesided: bool = True
    time_dim: int = 0
    channel_dim: int = -1
    length_key: str = "_emg_stft_length"
    cleanup_length_key: bool = True

    def __post_init__(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("n_fft must be > 0.")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be > 0.")
        if self.win_length is not None and self.win_length <= 0:
            raise ValueError("win_length must be > 0 when provided.")

    def __call__(
        self, data: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor | dict[str, Any]:
        length: int | None = None
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            length_value = data.get(self.length_key)
            if isinstance(length_value, int):
                length = length_value
            data["emg"] = self._invert(emg, length=length)
            if self.cleanup_length_key:
                data.pop(self.length_key, None)
            return data

        return self._invert(data, length=length)

    def _invert(self, spec: torch.Tensor, length: int | None) -> torch.Tensor:
        if spec.ndim not in (2, 3):
            raise ValueError(
                f"Expected spectrogram with 2 or 3 dims, got {tuple(spec.shape)}"
            )

        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        win_length = self.win_length if self.win_length is not None else self.n_fft
        # Ensure window is created on the same device as the input tensor
        window = (
            torch.hann_window(win_length, device=spec.device, dtype=spec.real.dtype)
            if self.window == "hann"
            else None
        )
        if window is None:
            raise ValueError(f"Unsupported window type: {self.window}")

        emg_ct = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=win_length,
            window=window,
            center=self.center,
            onesided=self.onesided,
            length=length,
        )

        emg = emg_ct.transpose(0, 1).contiguous()
        time_dim = self.time_dim if self.time_dim >= 0 else emg.ndim + self.time_dim
        channel_dim = (
            self.channel_dim if self.channel_dim >= 0 else emg.ndim + self.channel_dim
        )
        if {time_dim, channel_dim} != {0, 1}:
            emg = emg.movedim((0, 1), (time_dim, channel_dim))

        if squeezed:
            return emg
        return emg

@dataclass
class ExtractField:
    """从 NumPy 结构化数组中提取指定字段（如 'emg'）。"""
    field: str = "emg"

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # data 原本是结构化数组，data[self.field] 返回纯数值数组
        return np.ascontiguousarray(data[self.field])

@dataclass
class ToFloatTensor:
    """简单的 NumPy 到 Tensor 转换。"""
    def __call__(
        self, data: np.ndarray | dict[str, Any]
    ) -> torch.Tensor | dict[str, Any]:
        if isinstance(data, dict):
            for key in ("emg", "joint_angles"):
                if key in data and isinstance(data[key], np.ndarray):
                    data[key] = torch.from_numpy(data[key]).float()
            return data
        return torch.from_numpy(data).float()

@dataclass
class ExtractStructuredFields:
    """Extract fields from a structured array into a dict.

    If input is already a dict, it is returned as-is. If input is a plain
    ndarray, it is returned under the "emg" key for backward compatibility.
    """

    emg_field: str = "emg"
    joint_angles_field: str = "joint_angles"

    def __call__(
        self, data: np.ndarray | dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(data, dict):
            return data
        if isinstance(data, np.ndarray) and data.dtype.fields is not None:
            emg = np.ascontiguousarray(data[self.emg_field])
            joint_angles = np.ascontiguousarray(data[self.joint_angles_field])
            return {"emg": emg, "joint_angles": joint_angles}
        return {"emg": data}

@dataclass
class ExtractToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    field: str = "emg"

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data[self.field])


@dataclass
class ChannelDownsampling:
    """Downsample number of emg channels."""

    downsampling: int = 2

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[:, :: self.downsampling]


@dataclass
class ChannelSelect:
    """Select a fixed subset of EMG channels by index."""

    indices: Sequence[int]

    def __post_init__(self):
        if len(self.indices) == 0:
            raise ValueError("indices must contain at least one channel index.")
        self._indices = list(self.indices)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[..., self._indices]


@dataclass
class RandomChannelMask:
    """Randomly mask entire channels by zeroing them."""

    mask_prob: float = 0.0
    min_masked: int = 0
    max_masked: int | None = None
    channel_dim: int = -1
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1].")
        if self.min_masked < 0:
            raise ValueError("min_masked must be >= 0.")
        if self.max_masked is not None and self.max_masked < self.min_masked:
            raise ValueError("max_masked must be >= min_masked.")

    def __call__(
        self, data: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data
        if self.mask_prob <= 0.0 and self.min_masked == 0:
            return data

        n_channels = data.shape[self.channel_dim]
        if n_channels <= 0:
            return data

        mask = np.random.rand(n_channels) < self.mask_prob
        mask_indices = np.flatnonzero(mask)

        if self.max_masked is not None and mask_indices.size > self.max_masked:
            mask_indices = np.random.choice(
                mask_indices, size=self.max_masked, replace=False
            )

        if mask_indices.size < self.min_masked:
            available = np.setdiff1d(np.arange(n_channels), mask_indices)
            if available.size > 0:
                need = min(self.min_masked - mask_indices.size, available.size)
                extra = np.random.choice(available, size=need, replace=False)
                mask_indices = np.concatenate([mask_indices, extra], axis=0)

        if mask_indices.size == 0:
            return data

        masked = data.clone()
        index = torch.as_tensor(mask_indices, dtype=torch.long, device=masked.device)
        return masked.index_fill(self.channel_dim, index, self.mask_value)


@dataclass
class RandomTimeMask:
    """Randomly mask contiguous time segments across all channels."""

    max_mask_size: int
    num_masks: int = 1
    min_mask_size: int = 0
    time_dim: int = 0
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        if self.max_mask_size < 0:
            raise ValueError("max_mask_size must be >= 0.")
        if self.min_mask_size < 0:
            raise ValueError("min_mask_size must be >= 0.")
        if self.max_mask_size < self.min_mask_size:
            raise ValueError("max_mask_size must be >= min_mask_size.")
        if self.num_masks < 0:
            raise ValueError("num_masks must be >= 0.")

    def __call__(
        self, data: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data
        if self.max_mask_size == 0 or self.num_masks == 0:
            return data

        n_time = data.shape[self.time_dim]
        if n_time <= 0:
            return data

        masked = data.clone()
        for _ in range(self.num_masks):
            if self.max_mask_size == self.min_mask_size:
                mask_size = self.max_mask_size
            else:
                mask_size = np.random.randint(
                    self.min_mask_size, self.max_mask_size + 1
                )
            mask_size = min(mask_size, n_time)
            if mask_size == 0:
                continue
            start = np.random.randint(0, n_time - mask_size + 1)
            index = torch.arange(
                start,
                start + mask_size,
                device=masked.device,
                dtype=torch.long,
            )
            masked.index_fill_(self.time_dim, index, self.mask_value)

        return masked


@dataclass
class RandomFrequencyMask:
    """Randomly mask contiguous frequency bands via rFFT along time dimension."""

    max_mask_size: int
    num_masks: int = 1
    min_mask_size: int = 0
    time_dim: int = 0

    def __post_init__(self) -> None:
        if self.max_mask_size < 0:
            raise ValueError("max_mask_size must be >= 0.")
        if self.min_mask_size < 0:
            raise ValueError("min_mask_size must be >= 0.")
        if self.max_mask_size < self.min_mask_size:
            raise ValueError("max_mask_size must be >= min_mask_size.")
        if self.num_masks < 0:
            raise ValueError("num_masks must be >= 0.")

    def __call__(
        self, data: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data
        if self.max_mask_size == 0 or self.num_masks == 0:
            return data

        n_time = data.shape[self.time_dim]
        if n_time <= 0:
            return data

        spec = torch.fft.rfft(data, dim=self.time_dim)
        n_freq = spec.shape[self.time_dim]

        for _ in range(self.num_masks):
            if self.max_mask_size == self.min_mask_size:
                mask_size = self.max_mask_size
            else:
                mask_size = np.random.randint(
                    self.min_mask_size, self.max_mask_size + 1
                )
            mask_size = min(mask_size, n_freq)
            if mask_size == 0:
                continue
            start = np.random.randint(0, n_freq - mask_size + 1)
            index = torch.arange(
                start,
                start + mask_size,
                device=spec.device,
                dtype=torch.long,
            )
            spec.index_fill_(self.time_dim, index, 0)

        return torch.fft.irfft(spec, n=n_time, dim=self.time_dim)

@dataclass
class RandomGaussianNoise:
    """基于信噪比(SNR)随机添加高斯噪声 (支持 NumPy/Torch)"""
    min_snr_db: float = 25.0
    max_snr_db: float = 35.0
    apply_prob: float = 0.5
    time_dim: int = 0

    def __post_init__(self) -> None:
        if self.max_snr_db <= self.min_snr_db:
            raise ValueError("max_snr_db must be > min_snr_db.")
        if not 0.0 <= self.apply_prob <= 1.0:
            raise ValueError("apply_prob must be in [0, 1].")

    def __call__(
        self, data: np.ndarray | torch.Tensor | dict[str, Any]
    ) -> np.ndarray | torch.Tensor | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data
        if isinstance(data, torch.Tensor):
            if self.apply_prob == 0.0:
                return data
            if torch.rand((), device=data.device).item() > self.apply_prob:
                return data

            signal_power = data.pow(2).mean()
            if signal_power.item() == 0.0:
                return data

            snr = torch.empty((), device=data.device).uniform_(
                self.min_snr_db, self.max_snr_db
            )
            noise_power = signal_power / (10 ** (snr / 10))
            noise_std = noise_power.sqrt()
            noise = torch.randn_like(data) * noise_std
            return data + noise

        if self.apply_prob == 0.0 or np.random.rand() > self.apply_prob:
            return data

        signal_power = np.mean(data**2)
        if signal_power == 0:
            return data

        snr = np.random.uniform(self.min_snr_db, self.max_snr_db)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)

        noise = np.random.normal(0, noise_std, size=data.shape)
        return data + noise


@dataclass
class RandomMagnitudeWarping:
    """通过低频平滑曲线随机扭曲信号幅值 (NumPy 版)"""
    sigma: float = 0.15
    num_knots: int = 12
    mask_prob: float = 0.5
    time_dim: int = 0
    channel_independent: bool = True

    def __call__(
        self, data: np.ndarray | dict[str, Any]
    ) -> np.ndarray | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data
        if (
            self.mask_prob == 0.0
            or self.sigma == 0
            or np.random.rand() > self.mask_prob
        ):
            return data

        # 统一转为 (Channel, Time) 进行内部处理
        if self.time_dim == 0:
            x = data.T  # (C, T)
        else:
            x = data

        C, T = x.shape
        time_points = np.arange(T)
        knots_x = np.linspace(0, T - 1, self.num_knots)
        
        if self.channel_independent:
            knots_y = np.random.normal(
                loc=1.0, scale=self.sigma, size=(C, self.num_knots)
            )
            warping_curves = np.zeros((C, T))
            for i in range(C):
                cs = CubicSpline(knots_x, knots_y[i])
                warping_curves[i] = cs(time_points)
        else:
            knots_y = np.random.normal(
                loc=1.0, scale=self.sigma, size=(self.num_knots,)
            )
            cs = CubicSpline(knots_x, knots_y)
            warping_curves = cs(time_points)[np.newaxis, :]

        out = x * warping_curves
        return out.T if self.time_dim == 0 else out


@dataclass
class BaselineDrift:
    """叠加低频基线漂移（正弦波或随机行走噪声）(NumPy 版)"""

    sample_rate: float = 2000.0
    min_freq: float = 0.05
    max_freq: float = 0.5
    min_amp_ratio: float = 0.02
    max_amp_ratio: float = 0.08
    walk_step_std_ratio: float = 0.002
    mode: str = "both"  # "sine", "random_walk", "both"
    mask_prob: float = 0.3
    time_dim: int = 0
    channel_independent: bool = True

    def __call__(
        self, data: np.ndarray | dict[str, Any]
    ) -> np.ndarray | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data
        if self.mask_prob == 0.0 or np.random.rand() > self.mask_prob:
            return data

        if self.time_dim == 0:
            x = data.T  # (C, T)
        else:
            x = data

        c, t = x.shape
        if t == 0:
            return data
        time = np.arange(t) / float(self.sample_rate)
        use_sine = self.mode in {"sine", "both"}
        use_walk = self.mode in {"random_walk", "both"}

        if self.channel_independent:
            scale = np.std(x, axis=1, keepdims=True)
        else:
            scale = np.full((1, 1), np.std(x))

        drift = np.zeros_like(x, dtype=float)
        if use_sine:
            freq = np.random.uniform(self.min_freq, self.max_freq)
            phase = np.random.uniform(
                0.0,
                2.0 * np.pi,
                size=(c, 1) if self.channel_independent else (1, 1),
            )
            amp = np.random.uniform(
                self.min_amp_ratio,
                self.max_amp_ratio,
                size=(c, 1) if self.channel_independent else (1, 1),
            )
            drift += (scale * amp) * np.sin(2.0 * np.pi * freq * time + phase)

        if use_walk:
            step_std = self.walk_step_std_ratio * scale
            steps = np.random.normal(0.0, step_std, size=(c, t))
            walk = np.cumsum(steps, axis=1)
            walk -= np.mean(walk, axis=1, keepdims=True)
            drift += walk

        out = x + drift
        return out.T if self.time_dim == 0 else out


@dataclass
class PowerlineNoise:
    """注入工频干扰（50/60Hz 及其谐波）(NumPy 版)"""

    sample_rate: float = 2000.0
    base_freqs: Sequence[float] = (50.0, 60.0)
    max_harmonic: int = 3
    min_amp_ratio: float = 0.005
    max_amp_ratio: float = 0.03
    mask_prob: float = 0.3
    time_dim: int = 0
    channel_independent: bool = True

    def __call__(
        self, data: np.ndarray | dict[str, Any]
    ) -> np.ndarray | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data
        if self.mask_prob == 0.0 or np.random.rand() > self.mask_prob:
            return data

        if self.time_dim == 0:
            x = data.T  # (C, T)
        else:
            x = data

        c, t = x.shape
        if t == 0:
            return data
        time = np.arange(t) / float(self.sample_rate)
        nyquist = 0.5 * float(self.sample_rate)
        base = float(np.random.choice(self.base_freqs))

        if self.channel_independent:
            scale = np.std(x, axis=1, keepdims=True)
        else:
            scale = np.full((1, 1), np.std(x))

        noise = np.zeros_like(x, dtype=float)
        for harmonic in range(1, self.max_harmonic + 1):
            freq = base * harmonic
            if freq >= nyquist:
                break
            phase = np.random.uniform(
                0.0,
                2.0 * np.pi,
                size=(c, 1) if self.channel_independent else (1, 1),
            )
            amp = np.random.uniform(
                self.min_amp_ratio,
                self.max_amp_ratio,
                size=(c, 1) if self.channel_independent else (1, 1),
            )
            noise += (scale * amp) * np.sin(2.0 * np.pi * freq * time + phase)

        out = x + noise
        return out.T if self.time_dim == 0 else out


@dataclass
class LocalTimeWarp:
    """局部时间扭曲：基于 PCHIP 的时间索引重采样 (NumPy 版)"""

    min_num_knots: int = 5
    max_num_knots: int = 8
    max_jitter_ratio: float = 0.2
    mask_prob: float = 0.3
    time_dim: int = 0

    def __call__(
        self, data: np.ndarray | dict[str, Any]
    ) -> dict[str, Any] | np.ndarray:
        emg, joint_angles = self._extract_arrays(data)
        if emg is None:
            return data
        if self.mask_prob == 0.0 or np.random.rand() > self.mask_prob:
            return self._repack(data, emg, joint_angles)

        warp_idx = self._build_warp_indices(emg.shape[self.time_dim])
        warped_emg = self._warp_array(emg, warp_idx)
        warped_joint = None
        if joint_angles is not None:
            warped_joint = self._warp_array(joint_angles, warp_idx)
        return self._repack(data, warped_emg, warped_joint)

    def _extract_arrays(
        self, data: np.ndarray | dict[str, Any]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if isinstance(data, dict):
            return data.get("emg"), data.get("joint_angles")
        if isinstance(data, np.ndarray) and data.dtype.fields is not None:
            return data["emg"], data["joint_angles"]
        return None, None

    def _repack(
        self,
        data: np.ndarray | dict[str, Any],
        emg: np.ndarray,
        joint_angles: np.ndarray | None,
    ) -> dict[str, Any] | np.ndarray:
        if isinstance(data, dict):
            data["emg"] = emg
            if joint_angles is not None:
                data["joint_angles"] = joint_angles
            return data
        if isinstance(data, np.ndarray) and data.dtype.fields is not None:
            return {"emg": emg, "joint_angles": joint_angles}
        return {"emg": emg, "joint_angles": joint_angles}

    def _build_warp_indices(self, length: int) -> np.ndarray:
        if length <= 1:
            return np.arange(length, dtype=float)
        num_knots = np.random.randint(self.min_num_knots, self.max_num_knots + 1)
        base = np.linspace(0, length - 1, num_knots)
        seg = (length - 1) / max(num_knots - 1, 1)
        jitter = (
            np.random.uniform(
                -self.max_jitter_ratio, self.max_jitter_ratio, size=num_knots
            )
            * seg
        )
        warped = np.clip(base + jitter, 0, length - 1)
        warped[0] = 0.0
        warped[-1] = float(length - 1)
        warped = np.maximum.accumulate(warped)
        if not np.all(np.diff(warped) > 0):
            warped = np.linspace(0, length - 1, num_knots)
        warp_fn = PchipInterpolator(base, warped)
        return np.clip(warp_fn(np.arange(length)), 0, length - 1)

    def _warp_array(self, data: np.ndarray, warp_idx: np.ndarray) -> np.ndarray:
        x = np.moveaxis(data, self.time_dim, 0)
        t = np.arange(x.shape[0])
        warped = np.empty_like(x)
        for i in range(x.shape[1]):
            warped[:, i] = np.interp(warp_idx, t, x[:, i])
        return np.moveaxis(warped, 0, self.time_dim)

@dataclass
class RotationAugmentation:
    """旋转通道 (NumPy 版)"""
    channel_dim: int = -1

    def __call__(
        self, data: np.ndarray | dict[str, Any]
    ) -> np.ndarray | dict[str, Any]:
        if isinstance(data, dict):
            emg = data.get("emg")
            if emg is None:
                return data
            data["emg"] = self(emg)
            return data
        rotation = np.random.choice([-1, 0, 1])
        if rotation == 0:
            return data
        if torch.is_tensor(data):
            return torch.roll(data, shifts=rotation, dims=self.channel_dim)
        return np.roll(data, rotation, axis=self.channel_dim)


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data
