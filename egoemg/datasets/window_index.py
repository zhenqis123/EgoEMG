"""Window indices for session-based pretraining corpora."""

import numpy as np


def build_window_blocks(session_lengths, allowed_sessions, window_length, stride):
    """Index windows within selected sessions, preserving their order."""
    block_session_idx: list[int] = []
    block_start: list[int] = []
    block_end: list[int] = []
    block_lengths: list[int] = []

    for si in allowed_sessions:
        slen = int(session_lengths[si])
        if slen < window_length:
            continue
        n = (slen - window_length) // stride + 1
        block_session_idx.append(int(si))
        block_start.append(0)
        block_end.append(slen)
        block_lengths.append(n)

    block_session_idx = np.asarray(block_session_idx, dtype=np.int32)
    block_start = np.asarray(block_start, dtype=np.int64)
    block_end = np.asarray(block_end, dtype=np.int64)
    block_cumsum = np.cumsum(np.asarray([0] + block_lengths, dtype=np.int64))
    return block_session_idx, block_start, block_end, block_cumsum
