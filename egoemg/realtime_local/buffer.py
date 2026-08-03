from __future__ import annotations

import numpy as np


class SlidingWindowBuffer:
    """Fixed-size ring buffer that emits the latest window on a sample stride."""

    def __init__(self, window_length: int, stride: int, n_channels: int = 8) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if stride > window_length:
            raise ValueError("stride must be <= window_length")
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.n_channels = int(n_channels)
        self._buf = np.zeros((self.window_length, self.n_channels), dtype=np.float32)
        self._write_pos = 0
        self._last_emit_pos = 0

    def push(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.ndim != 2 or samples.shape[1] != self.n_channels:
            raise ValueError(f"Expected (N, {self.n_channels}), got {samples.shape}")

        n = samples.shape[0]
        if n >= self.window_length:
            self._buf[:] = samples[-self.window_length :]
            self._write_pos += n
            return

        start = self._write_pos % self.window_length
        end = start + n
        if end <= self.window_length:
            self._buf[start:end] = samples
        else:
            first = self.window_length - start
            self._buf[start:] = samples[:first]
            self._buf[: end % self.window_length] = samples[first:]
        self._write_pos += n

    def ready_count(self) -> int:
        if self._write_pos < self.window_length:
            return 0
        return max(0, (self._write_pos - self._last_emit_pos) // self.stride)

    def has_window(self) -> bool:
        return self.ready_count() > 0

    def get_window(self) -> np.ndarray:
        if self._write_pos < self.window_length:
            raise RuntimeError("Not enough samples collected yet")
        self._last_emit_pos = self._write_pos
        start = self._write_pos - self.window_length
        idx = np.arange(start, self._write_pos) % self.window_length
        return self._buf[idx].copy()

    def skip_to_latest(self) -> None:
        """Drop pending emits while keeping buffered samples."""
        self._last_emit_pos = self._write_pos

    def keep_latest_ready(self) -> None:
        """Drop stale emits but leave one latest window ready to consume."""
        if self._write_pos >= self.window_length:
            self._last_emit_pos = max(0, self._write_pos - self.stride)

    @property
    def total_samples(self) -> int:
        return self._write_pos

    @property
    def is_full(self) -> bool:
        return self._write_pos >= self.window_length
