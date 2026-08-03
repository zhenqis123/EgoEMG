from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi
except ImportError as exc:  # pragma: no cover - exercised by deployment env
    raise RuntimeError("scipy is required for local realtime preprocessing") from exc


DEFAULT_FS_HZ = 2000.0
DEFAULT_NOISE_FLOOR = np.array(
    [
        0.20663154703820982,
        0.4506581124067573,
        0.1747040774784117,
        0.27043118610147415,
        0.25725022671584913,
        0.20483059055773223,
        0.1538460463585672,
        0.4449335286200831,
    ],
    dtype=np.float32,
)


def load_noise_floor(path: str | Path | None = None, hand: str = "right") -> np.ndarray:
    if path is None:
        return DEFAULT_NOISE_FLOOR.copy()
    with open(Path(path), encoding="utf-8") as f:
        data = json.load(f)
    if "noise_floor" in data:
        return np.asarray(data["noise_floor"], dtype=np.float32)
    item = data[hand]
    return np.asarray(item["noise_floor"], dtype=np.float32)


class OnlineSosFilter:
    """Stateful causal SOS filter for sample blocks shaped (N, C)."""

    def __init__(self, sos: np.ndarray, n_channels: int) -> None:
        self.sos = sos
        self.n_channels = n_channels
        self.zi_template = sosfilt_zi(sos).astype(np.float64)
        self.zi = np.repeat(self.zi_template[:, :, None], n_channels, axis=2)
        self.initialized = False

    def reset(self) -> None:
        self.zi = np.repeat(self.zi_template[:, :, None], self.n_channels, axis=2)
        self.initialized = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        if x64.ndim != 2 or x64.shape[1] != self.n_channels:
            raise ValueError(f"Expected (N, {self.n_channels}), got {x64.shape}")
        if x64.shape[0] == 0:
            return x64.astype(np.float32)
        if not self.initialized:
            self.zi = self.zi_template[:, :, None] * x64[0][None, None, :]
            self.initialized = True
        y, self.zi = sosfilt(self.sos, x64, axis=0, zi=self.zi)
        return y.astype(np.float32)


class SmallPreprocessor:
    """Online approximation of the legacy ``emg_*_filtered_aligned`` contract.

    NOTE: ``filtered_aligned`` data and its offline generation pipeline have been
    removed (see unified_filter_align.py, now deleted). Active training uses
    ``filtered_paper`` instead. This class is kept for backward compatibility
    with models trained on filtered_aligned; for filtered_paper-based models,
    use the FFT-domain bandpass + notch pipeline instead.

    The legacy field was produced offline with zero-phase Butterworth filters
    and per-channel noise-floor rescaling. This class uses causal SOS filters
    with persistent state for realtime operation.
    """

    def __init__(
        self,
        n_channels: int = 8,
        fs_hz: float = DEFAULT_FS_HZ,
        noise_floor: np.ndarray | None = None,
        input_scale: float = 1.0,
        remove_sample_mean: bool = False,
    ) -> None:
        self.n_channels = int(n_channels)
        self.fs_hz = float(fs_hz)
        self.noise_floor = (
            np.asarray(noise_floor, dtype=np.float32)
            if noise_floor is not None
            else DEFAULT_NOISE_FLOOR.copy()
        )
        if self.noise_floor.shape != (self.n_channels,):
            raise ValueError(
                f"noise_floor must have shape ({self.n_channels},), got {self.noise_floor.shape}"
            )
        self.input_scale = float(input_scale)
        self.remove_sample_mean = bool(remove_sample_mean)

        self.hp20 = OnlineSosFilter(
            butter(4, 20.0, btype="highpass", fs=self.fs_hz, output="sos"),
            self.n_channels,
        )
        self.lp850 = OnlineSosFilter(
            butter(4, 850.0, btype="lowpass", fs=self.fs_hz, output="sos"),
            self.n_channels,
        )
        self.hp40 = OnlineSosFilter(
            butter(4, 40.0, btype="highpass", fs=self.fs_hz, output="sos"),
            self.n_channels,
        )

    def reset(self) -> None:
        self.hp20.reset()
        self.lp850.reset()
        self.hp40.reset()

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            raise ValueError(f"Expected (N, {self.n_channels}), got {x.shape}")
        if self.remove_sample_mean:
            x = x - x.mean(axis=1, keepdims=True)
        x = x * self.input_scale
        x = self.hp20(x)
        x = self.lp850(x)
        x = self.hp40(x)
        denom = np.where(self.noise_floor < 1e-8, 1.0, self.noise_floor)
        return (x / denom[None, :]).astype(np.float32)
