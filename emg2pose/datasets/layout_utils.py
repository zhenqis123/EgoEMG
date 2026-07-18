from __future__ import annotations

import numpy as np

_CIRCULAR_INTERP_CACHE: dict[tuple[int, int], np.ndarray] = {}
_SPARSE_RING_INTERP_CACHE: dict[tuple[tuple[int, ...], int], np.ndarray] = {}


def get_circular_interp_matrix(src_channels: int, target_channels: int) -> np.ndarray:
    key = (src_channels, target_channels)
    cached = _CIRCULAR_INTERP_CACHE.get(key)
    if cached is not None:
        return cached

    pos = np.arange(target_channels, dtype=np.float32) * (src_channels / target_channels)
    idx0 = np.floor(pos).astype(np.int64) % src_channels
    idx1 = (idx0 + 1) % src_channels
    w1 = (pos - np.floor(pos)).astype(np.float32)
    w0 = (1.0 - w1).astype(np.float32)
    weights = np.zeros((src_channels, target_channels), dtype=np.float32)
    weights[idx0, np.arange(target_channels)] = w0
    weights[idx1, np.arange(target_channels)] += w1
    _CIRCULAR_INTERP_CACHE[key] = weights
    return weights


def circular_interpolate(data: np.ndarray, target_channels: int) -> np.ndarray:
    """Linear circular interpolation with cached weight matrix.

    Args:
        data: shape (T, C)
        target_channels: output channels
    """
    _, src_channels = data.shape
    weights = get_circular_interp_matrix(src_channels, target_channels)
    return data @ weights


def place_sparse_channels(
    data: np.ndarray,
    target_channels: int,
    target_positions: np.ndarray,
) -> np.ndarray:
    """Place source channels into selected target positions and zero-fill the rest.

    Args:
        data: shape (T, C)
        target_channels: output channel count
        target_positions: 0-based target positions, shape (C,)
    """
    out = np.zeros((data.shape[0], target_channels), dtype=np.float32)
    out[:, target_positions] = data.astype(np.float32, copy=False)
    return out


def get_sparse_ring_interp_matrix(
    target_positions: np.ndarray,
    target_channels: int,
) -> np.ndarray:
    """Interpolate sparse channel anchors onto a full circular layout.

    Args:
        target_positions: 0-based anchor positions on the output ring, shape (K,)
        target_channels: total output channels on the ring

    Returns:
        weights of shape (K, target_channels)
    """
    key = (tuple(int(x) for x in target_positions.tolist()), int(target_channels))
    cached = _SPARSE_RING_INTERP_CACHE.get(key)
    if cached is not None:
        return cached

    pos = np.asarray(target_positions, dtype=np.int64)
    if pos.ndim != 1 or len(pos) == 0:
        raise ValueError("target_positions must be a non-empty 1D array")
    if np.any(pos < 0) or np.any(pos >= target_channels):
        raise ValueError("target_positions out of range")
    if len(np.unique(pos)) != len(pos):
        raise ValueError("target_positions must be unique")

    order = np.argsort(pos)
    sorted_pos = pos[order]
    k = len(sorted_pos)
    weights_sorted = np.zeros((k, target_channels), dtype=np.float32)

    for out_idx in range(target_channels):
        if out_idx in sorted_pos:
            anchor_idx = int(np.where(sorted_pos == out_idx)[0][0])
            weights_sorted[anchor_idx, out_idx] = 1.0
            continue

        insert = int(np.searchsorted(sorted_pos, out_idx, side="right"))
        right_i = insert % k
        left_i = (insert - 1) % k
        left_pos = int(sorted_pos[left_i])
        right_pos = int(sorted_pos[right_i])
        gap = (right_pos - left_pos) % target_channels
        if gap == 0:
            weights_sorted[left_i, out_idx] = 1.0
            continue
        d_left = (out_idx - left_pos) % target_channels
        d_right = (right_pos - out_idx) % target_channels
        weights_sorted[left_i, out_idx] = np.float32(d_right / gap)
        weights_sorted[right_i, out_idx] = np.float32(d_left / gap)

    inv_order = np.argsort(order)
    weights = weights_sorted[inv_order]
    _SPARSE_RING_INTERP_CACHE[key] = weights
    return weights


def sparse_ring_interpolate(
    data: np.ndarray,
    target_channels: int,
    target_positions: np.ndarray,
) -> np.ndarray:
    """Interpolate sparse anchor channels to a dense circular target layout.

    Args:
        data: shape (T, C)
        target_channels: number of output channels
        target_positions: 0-based anchor positions for the C source channels
    """
    if data.shape[1] != len(target_positions):
        raise ValueError(
            f"data has {data.shape[1]} channels but target_positions has {len(target_positions)} anchors"
        )
    weights = get_sparse_ring_interp_matrix(target_positions, target_channels)
    return data @ weights
