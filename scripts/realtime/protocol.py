"""Wire format encode/decode for ZMQ EMG streaming.

Upstream (client -> server): binary frames with raw EMG samples.
  Layout: [uint64 seq][uint64 ts_ns][uint32 n_samples][float32[n*8] emg]

Downstream (server -> client): JSON frames with predictions.
  Layout: {"seq", "timestamp_ns", "angles": [...], "landmarks": [...], "inference_ms"}
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

# Upstream header: seq(u64) + ts_ns(u64) + n_samples(u32) = 20 bytes
_HEADER_FMT = "<QQI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 20 bytes
_N_CHANNELS = 8


def encode_upstream(
    seq: int,
    timestamp_ns: int,
    samples: np.ndarray,
) -> bytes:
    """Encode a batch of EMG samples into a binary frame.

    Args:
        seq: Sequence number for ordering.
        timestamp_ns: Client clock timestamp in nanoseconds.
        samples: (n_samples, 8) float32 array of EMG values.

    Returns:
        Binary frame ready for ZMQ send.
    """
    n_samples = samples.shape[0]
    header = struct.pack(_HEADER_FMT, seq, timestamp_ns, n_samples)
    # Ensure contiguous float32
    body = np.ascontiguousarray(samples, dtype=np.float32).tobytes()
    return header + body


def decode_upstream(frame: bytes) -> tuple[int, int, np.ndarray]:
    """Decode a binary upstream frame.

    Returns:
        (seq, timestamp_ns, samples) where samples is (n, 8) float32.
    """
    seq, ts_ns, n_samples = struct.unpack(_HEADER_FMT, frame[:_HEADER_SIZE])
    body = frame[_HEADER_SIZE:]
    samples = np.frombuffer(body, dtype=np.float32).reshape(n_samples, _N_CHANNELS)
    return seq, ts_ns, samples


def encode_downstream(
    seq: int,
    timestamp_ns: int,
    angles: np.ndarray,
    landmarks: np.ndarray | None = None,
    inference_ms: float = 0.0,
) -> dict[str, Any]:
    """Encode a prediction result as a JSON-serializable dict.

    Args:
        seq: Prediction sequence number.
        timestamp_ns: Timestamp of the last sample in the window.
        angles: (22,) or (20,) array of joint angles in radians.
        landmarks: (21, 3) array of hand keypoints (optional).
        inference_ms: Inference time in milliseconds.

    Returns:
        Dict ready for json.dumps / zmq send_json.
    """
    result: dict[str, Any] = {
        "seq": seq,
        "timestamp_ns": timestamp_ns,
        "angles": angles.tolist(),
        "inference_ms": round(inference_ms, 2),
    }
    if landmarks is not None:
        result["landmarks"] = landmarks.tolist()
    return result
