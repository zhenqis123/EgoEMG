"""Real-time FFT filter matching the offline EMG filtering pipeline.

Contains the same frequency mask construction as filter_emg_into_new_columns.py
(bandpass 20-850 Hz + notch 50/100 Hz) but without the pyarrow dependency,
so it can run in lightweight client environments.
"""

from __future__ import annotations

import numpy as np

# Filter parameters (must match filter_emg_into_new_columns.py exactly)
_FS = 2000.0
_LOW_CUT = 20.0
_LOW_TRANSITION = 5.0
_HIGH_CUT = 850.0
_HIGH_TRANSITION = 50.0
_NOTCH_CONFIGS = (
    {"center": 50.0, "stop_half_width": 1.5, "transition_half_width": 1.5},
    {"center": 100.0, "stop_half_width": 1.5, "transition_half_width": 1.5},
)


def _smoothstep_cosine(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def _build_emg_frequency_mask(n_samples: int, fs: float = _FS) -> np.ndarray:
    """Build the frequency-domain mask for EMG filtering.

    Identical to filter_emg_into_new_columns.build_emg_frequency_mask.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = np.ones_like(freqs, dtype=np.float64)

    # High-pass with soft roll-off
    hp0 = max(0.0, _LOW_CUT - _LOW_TRANSITION)
    hp1 = _LOW_CUT
    if hp1 > hp0:
        below = freqs <= hp0
        trans = (freqs > hp0) & (freqs < hp1)
        mask[below] = 0.0
        mask[trans] *= _smoothstep_cosine((freqs[trans] - hp0) / (hp1 - hp0))

    # Low-pass with soft roll-off
    lp0 = _HIGH_CUT
    lp1 = min(fs * 0.5, _HIGH_CUT + _HIGH_TRANSITION)
    if lp1 > lp0:
        above = freqs >= lp1
        trans = (freqs > lp0) & (freqs < lp1)
        mask[above] = 0.0
        mask[trans] *= (1.0 - _smoothstep_cosine((freqs[trans] - lp0) / (lp1 - lp0)))

    # Notch filters for 50 Hz and 100 Hz mains harmonics
    for cfg in _NOTCH_CONFIGS:
        center = cfg["center"]
        stop_hw = cfg["stop_half_width"]
        trans_hw = cfg["transition_half_width"]
        stop_lo = center - stop_hw
        stop_hi = center + stop_hw
        trans_lo = stop_lo - trans_hw
        trans_hi = stop_hi + trans_hw

        hard = (freqs >= stop_lo) & (freqs <= stop_hi)
        left = (freqs > trans_lo) & (freqs < stop_lo)
        right = (freqs > stop_hi) & (freqs < trans_hi)
        mask[hard] = 0.0
        if np.any(left):
            mask[left] *= (
                1.0 - _smoothstep_cosine((freqs[left] - trans_lo) / (stop_lo - trans_lo))
            )
        if np.any(right):
            mask[right] *= _smoothstep_cosine(
                (freqs[right] - stop_hi) / (trans_hi - stop_hi)
            )

    return mask


def filter_emg_fft(x: np.ndarray, fs: float = _FS) -> np.ndarray:
    """Apply the canonical offline ``filtered_paper`` filter.

    This is the shared implementation for dataset conversion and realtime
    inference. Filtering is performed independently over the supplied episode
    or window, including per-channel mean subtraction.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (time, channels), got shape {x.shape}")
    if x.size == 0:
        return x.astype(np.float32, copy=True)
    x64 = x.astype(np.float64, copy=False)
    x0 = x64 - np.mean(x64, axis=0, keepdims=True)
    spectrum = np.fft.rfft(x0, axis=0)
    spectrum *= _build_emg_frequency_mask(x0.shape[0], fs=fs)[:, None]
    return np.fft.irfft(spectrum, n=x0.shape[0], axis=0).astype(np.float32)


class RealtimeFFTFilter:
    """Per-window FFT filter matching the offline pipeline exactly.

    Applies bandpass (20-850 Hz) + notch (50 Hz, 100 Hz) filtering to a
    fixed-length window of EMG data. The frequency mask is precomputed once
    at init time for efficiency.

    Args:
        window_length: Number of samples per window (e.g. 2000 for 1s at 2kHz).
        fs: Sampling rate in Hz.
    """

    def __init__(self, window_length: int, fs: float = _FS):
        self.window_length = window_length
        self.fs = fs
        # Pre-compute the frequency mask once: shape (n_freq, 1) for broadcasting
        mask_1d = _build_emg_frequency_mask(window_length, fs)
        self._mask = mask_1d[:, None].astype(np.float64)

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filter a (window_length, n_channels) EMG window.

        Steps (matching filter_emg_fft exactly):
          1. Subtract per-channel mean (DC removal)
          2. rfft along time axis
          3. Multiply by precomputed frequency mask
          4. irfft back to time domain

        Args:
            x: (window_length, n_channels) float32 array.

        Returns:
            Filtered array, same shape and dtype as input.
        """
        if x.shape[0] != self.window_length:
            raise ValueError(
                f"Expected {self.window_length} samples, got {x.shape[0]}"
            )

        x64 = x.astype(np.float64, copy=False)
        x0 = x64 - np.mean(x64, axis=0, keepdims=True)
        spectrum = np.fft.rfft(x0, axis=0)
        spectrum *= self._mask
        return np.fft.irfft(spectrum, n=x0.shape[0], axis=0).astype(np.float32)
