"""Sliding window ring buffer for accumulating streaming EMG samples."""

from __future__ import annotations

import numpy as np


class SlidingWindowBuffer:
    """Ring buffer that accumulates samples and emits fixed-length windows.

    Supports overlapping windows via configurable stride. For example, with
    window_length=2000 and stride=400, a new window is emitted every 400
    samples (200ms at 2kHz), with 80% overlap.

    Args:
        window_length: Number of samples per output window.
        stride: Number of new samples between consecutive windows.
        n_channels: Number of EMG channels (default 8).
    """

    def __init__(
        self,
        window_length: int,
        stride: int,
        n_channels: int = 8,
    ):
        if stride > window_length:
            raise ValueError(f"stride ({stride}) must be <= window_length ({window_length})")
        self.window_length = window_length
        self.stride = stride
        self.n_channels = n_channels

        self._buf = np.zeros((window_length, n_channels), dtype=np.float32)
        self._write_pos = 0  # total samples written (monotonic)
        self._last_emit_pos = 0  # _write_pos at which last window was emitted

    def push(self, samples: np.ndarray) -> None:
        """Push (n, n_channels) new samples into the ring buffer.

        Args:
            samples: (n, n_channels) float32 array.
        """
        if samples.ndim != 2 or samples.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected (n, {self.n_channels}), got {samples.shape}"
            )
        for i in range(samples.shape[0]):
            idx = self._write_pos % self.window_length
            self._buf[idx] = samples[i]
            self._write_pos += 1

    def has_window(self) -> bool:
        """Check if at least one new window is ready to emit."""
        if self._write_pos < self.window_length:
            return False
        return (self._write_pos - self._last_emit_pos) >= self.stride

    def get_window(self) -> np.ndarray:
        """Extract the latest window_length samples in chronological order.

        Returns:
            (window_length, n_channels) float32 array.
        """
        if self._write_pos < self.window_length:
            raise RuntimeError("Not enough samples collected yet")

        self._last_emit_pos = self._write_pos

        # Read from ring buffer in chronological order
        start = self._write_pos - self.window_length
        indices = np.arange(start, self._write_pos) % self.window_length
        return self._buf[indices].copy()

    @property
    def total_samples(self) -> int:
        """Total number of samples pushed so far."""
        return self._write_pos

    @property
    def is_full(self) -> bool:
        """Whether at least one full window has been accumulated."""
        return self._write_pos >= self.window_length
