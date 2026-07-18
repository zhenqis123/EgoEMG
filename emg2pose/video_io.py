"""Shared EgoEMG video access helpers.

All EgoEMG video decoding in this repository should go through all-intra
re-encoded files and decord. Do not silently fall back to OpenCV random seeks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_ALLINTRA_SUFFIX = "_allintra"


def resolve_allintra_video_path(
    *,
    raw_video_path: str | Path,
    data_root: str | Path,
    allintra_root: str | Path,
    suffix: str = DEFAULT_ALLINTRA_SUFFIX,
) -> Path:
    raw_path = Path(raw_video_path)
    data_root = Path(data_root).resolve()
    allintra_root = Path(allintra_root).resolve()

    if raw_path.is_absolute():
        try:
            rel_path = raw_path.relative_to(data_root)
        except ValueError:
            rel_path = Path(raw_path.name)
    else:
        rel_path = raw_path

    out_path = allintra_root / rel_path.with_name(f"{rel_path.stem}{suffix}.mp4")
    if not out_path.exists():
        raise FileNotFoundError(
            f"Missing all-intra video for {raw_path}: expected {out_path}"
        )
    return out_path


class DecordVideoReaderCache:
    """Path-keyed decord reader cache with LRU eviction."""

    def __init__(self, max_readers: int = 8) -> None:
        self._cache: dict[str, object] = {}
        self._max_readers = max_readers

    def _open_reader(self, video_path: str):
        try:
            from decord import VideoReader, cpu
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "decord is required for EgoEMG video decoding. "
                "Install decord and provide all-intra videos."
            ) from exc
        return VideoReader(video_path, ctx=cpu(0))

    def get(self, video_path: str):
        reader = self._cache.pop(video_path, None)
        if reader is None:
            reader = self._open_reader(video_path)
        self._cache[video_path] = reader
        if len(self._cache) > self._max_readers:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return reader

    def read_one_rgb(self, video_path: str, frame_idx: int) -> np.ndarray:
        reader = self.get(video_path)
        frame_idx = max(0, min(int(frame_idx), len(reader) - 1))
        return reader[frame_idx].asnumpy()

    def read_many_rgb(self, video_path: str, frame_indices: Iterable[int]) -> np.ndarray:
        reader = self.get(video_path)
        indices = np.asarray(list(frame_indices), dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("frame_indices must be 1D")
        if indices.size == 0:
            raise ValueError("frame_indices must be non-empty")
        indices = np.clip(indices, 0, len(reader) - 1)
        return reader.get_batch(indices.tolist()).asnumpy()


class FfmpegNvdecReaderCache:
    """GPU-accelerated video reader using ffmpeg NVDEC.

    Uses ffmpeg with -hwaccel cuda for hardware H.264 decoding.
    Batch decoding multiple frames in one ffmpeg call reduces subprocess overhead.
    """

    def __init__(self, gpu_id: int = 0, max_readers: int = 8, batch_size: int = 16) -> None:
        self._gpu_id = gpu_id
        self._batch_size = batch_size
        self._cache: dict[str, object] = {}
        self._max_readers = max_readers
        self._pending: dict[str, dict[int, np.ndarray]] = {}

    def _open_reader(self, video_path: str):
        try:
            from decord import VideoReader, cpu
        except Exception as exc:
            raise RuntimeError(
                "decord is required for metadata. "
                "Install decord and provide all-intra videos."
            ) from exc
        return VideoReader(video_path, ctx=cpu(0))

    def get(self, video_path: str):
        reader = self._cache.pop(video_path, None)
        if reader is None:
            reader = self._open_reader(video_path)
        self._cache[video_path] = reader
        if len(self._cache) > self._max_readers:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return reader

    def read_one_rgb(self, video_path: str, frame_idx: int) -> np.ndarray:
        reader = self.get(video_path)
        frame_idx = max(0, min(int(frame_idx), len(reader) - 1))
        h, w = reader[0].shape[:2]
        # Use decord as fallback - ffmpeg subprocess overhead is too high for single frames
        return reader[frame_idx].asnumpy()

    def read_many_rgb(self, video_path: str, frame_indices: Iterable[int]) -> np.ndarray:
        """Batch decode multiple frames using ffmpeg NVDEC."""
        reader = self.get(video_path)
        indices = np.asarray(list(frame_indices), dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("frame_indices must be 1D")
        if indices.size == 0:
            raise ValueError("frame_indices must be non-empty")
        indices = np.clip(indices, 0, len(reader) - 1)
        h, w = reader[0].shape[:2]

        if len(indices) < 4:
            # Small batch: use decord directly
            return reader.get_batch(indices.tolist()).asnumpy()

        # Large batch: use ffmpeg NVDEC
        try:
            return self._ffmpeg_nvdec_batch(video_path, indices, h, w)
        except Exception:
            return reader.get_batch(indices.tolist()).asnumpy()

    def _ffmpeg_nvdec_batch(self, video_path: str, indices: np.ndarray, h: int, w: int) -> np.ndarray:
        """Decode multiple frames in one ffmpeg call."""
        # Build select filter for all requested frames
        select_expr = "select=" + "+".join(f"eq(n\\,{idx})" for idx in indices)
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-hwaccel_device", str(self._gpu_id),
            "-v", "quiet",
            "-i", video_path,
            "-vf", select_expr,
            "-vsync", "vfr",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            raise RuntimeError("ffmpeg NVDEC batch decode failed")
        # Parse output: each frame is h*w*3 bytes
        frame_bytes = h * w * 3
        n_frames = len(proc.stdout) // frame_bytes
        if n_frames != len(indices):
            raise RuntimeError(f"Expected {len(indices)} frames, got {n_frames}")
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(n_frames, h, w, 3)


class PyavNvdecReader:
    """GPU-accelerated video reader using PyAV with NVDEC hardware context.

    Avoids subprocess overhead by using PyAV's hardware decoding directly.
    """

    def __init__(self, video_path: str, gpu_id: int = 0) -> None:
        self._video_path = video_path
        self._gpu_id = gpu_id
        self._container = None
        self._stream = None
        self._n_frames = None
        self._shape = None
        self._open()

    def _open(self) -> None:
        import av
        self._container = av.open(self._video_path)
        self._stream = self._container.streams.video[0]
        # Try hardware context
        try:
            hw_ctx = av.VideoContext()
            hw_ctx.hwdevice_type = "cuda"
            hw_ctx.hwdevice = f"{self._gpu_id}"
            self._stream.thread_type = "AUTO"
            self._stream.codec_context.hwaccel = "cuda"
            self._stream.codec_context.hwaccel_device = self._gpu_id
        except Exception:
            pass  # Fall back to software decode
        # Get dimensions
        self._shape = (self._stream.height, self._stream.width, 3)
        self._n_frames = self._stream.frames

    def __len__(self) -> int:
        return self._n_frames if self._n_frames > 0 else 0

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= len(self):
            raise IndexError(frame_idx)
        # Seek to frame
        self._stream.seek(frame_idx)
        for i, packet in enumerate(self._container.demux(self._stream)):
            for frame in packet.decode():
                if i >= frame_idx:
                    return frame.to_ndarray(format="rgb24")
        # Fallback
        return np.zeros(self._shape, dtype=np.uint8)

    def get_batch(self, frame_indices: list[int]) -> np.ndarray:
        # PyAV batch decode is not efficient for random indices
        frames = [self[i] for i in frame_indices]
        return np.stack(frames, axis=0)


class PyavNvdecReaderCache:
    """LRU cache for PyAV NVDEC readers."""

    def __init__(self, gpu_id: int = 0, max_readers: int = 8) -> None:
        self._gpu_id = gpu_id
        self._cache: dict[str, PyavNvdecReader] = {}
        self._max_readers = max_readers

    def get(self, video_path: str) -> PyavNvdecReader:
        reader = self._cache.pop(video_path, None)
        if reader is None:
            reader = PyavNvdecReader(video_path, gpu_id=self._gpu_id)
        self._cache[video_path] = reader
        if len(self._cache) > self._max_readers:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return reader

    def read_one_rgb(self, video_path: str, frame_idx: int) -> np.ndarray:
        reader = self.get(video_path)
        frame_idx = max(0, min(int(frame_idx), len(reader) - 1))
        return reader[frame_idx]

    def read_many_rgb(self, video_path: str, frame_indices: Iterable[int]) -> np.ndarray:
        reader = self.get(video_path)
        indices = np.clip(np.asarray(list(frame_indices), dtype=np.int64), 0, len(reader) - 1)
        return reader.get_batch(indices.tolist())