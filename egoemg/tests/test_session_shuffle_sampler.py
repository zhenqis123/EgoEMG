import numpy as np
from torch.utils.data.distributed import DistributedSampler

from egoemg.datamodule import SessionShuffleSampler

# Contiguous index ranges per session, mimicking a windowed memmap dataset.
GROUP_SIZES = [40, 10, 25, 5, 60, 30, 12]


def _make_groups():
    groups = []
    offset = 0
    for size in GROUP_SIZES:
        groups.append(np.arange(offset, offset + size))
        offset += size
    return groups


def _group_of_index(idx):
    return int(np.searchsorted(np.cumsum(GROUP_SIZES), idx, side="right"))


def test_is_distributed_sampler_subclass():
    # Lightning skips re-wrapping samplers that subclass DistributedSampler.
    s = SessionShuffleSampler(_make_groups(), batch_size=8)
    assert isinstance(s, DistributedSampler)


def test_single_rank_full_coverage():
    s = SessionShuffleSampler(
        _make_groups(), batch_size=8, block_batches=4, seed=3
    )
    out = np.fromiter(iter(s), dtype=np.int64)
    total = sum(GROUP_SIZES)
    assert len(s) == len(out) == 192  # 182 padded up to whole 32-blocks
    assert np.array_equal(np.unique(out), np.arange(total))  # every index covered


def test_multi_rank_balance_and_coverage():
    world = 3
    samplers = [
        SessionShuffleSampler(
            _make_groups(), batch_size=8, block_batches=4, seed=1, num_replicas=world, rank=r
        )
        for r in range(world)
    ]
    total = sum(GROUP_SIZES)
    outs = [np.fromiter(iter(s), dtype=np.int64) for s in samplers]
    assert all(len(o) == len(s) == 64 for o, s in zip(outs, samplers))  # equal shards
    union = np.unique(np.concatenate(outs))
    assert np.array_equal(union, np.arange(total))  # nothing lost across ranks


def test_reads_stay_block_sequential():
    # Core I/O property: every yielded block (block_size consecutive windows,
    # except a possibly shorter final block) touches each session in one
    # contiguous run, so disk access per block is a sequential sweep.
    batch_size, block_batches = 8, 4
    s = SessionShuffleSampler(
        _make_groups(), batch_size=batch_size, block_batches=block_batches, seed=5
    )
    out = np.fromiter(iter(s), dtype=np.int64)
    block = batch_size * block_batches
    starts = np.cumsum([0] + GROUP_SIZES)
    pos = 0
    while pos < len(out):
        size = min(block, len(out) - pos)  # final block may be shorter
        chunk = out[pos : pos + size]
        for g in range(len(GROUP_SIZES)):
            offs = chunk[(chunk >= starts[g]) & (chunk < starts[g + 1])] - starts[g]
            assert len(offs) <= 1 or (np.diff(np.sort(offs)) == 1).all(), (
                f"block at {pos} reads session {g} non-contiguously: {np.sort(offs)}"
            )
        pos += size


def test_determinism_and_epoch_variation():
    kw = dict(batch_size=8, seed=7, num_replicas=2, rank=0)
    s1 = SessionShuffleSampler(_make_groups(), **kw)
    s2 = SessionShuffleSampler(_make_groups(), **kw)
    o1 = list(iter(s1))
    assert o1 == list(iter(s2))  # same seed/epoch -> identical order

    s1.set_epoch(1)
    o2 = list(iter(s1))
    assert len(o1) == len(o2)  # same shard size
    assert o1 != o2  # but re-shuffled


def test_more_ranks_than_sessions_does_not_hang():
    groups = [np.arange(0, 10), np.arange(10, 18)]
    s = SessionShuffleSampler(
        groups, batch_size=4, block_batches=1, num_replicas=4, rank=2, seed=0
    )
    out = np.fromiter(iter(s), dtype=np.int64)
    assert len(out) == len(s) == 8  # per-rank 5 padded to whole 4-blocks
