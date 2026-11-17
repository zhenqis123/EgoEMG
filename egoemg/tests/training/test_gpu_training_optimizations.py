import pytest
import torch
from torch.nn.functional import interpolate
from torch.utils.data import TensorDataset

from egoemg.datasets.cuda_prefetch import CudaPrefetchDataLoader
from egoemg.models.modules.base import BaseModule


@pytest.mark.parametrize("shape", [(2, 13), (2, 22, 13)])
@pytest.mark.parametrize("new_length", [13, 7, 19])
def test_align_mask_exactly_matches_nearest_reference(shape, new_length):
    mask = torch.rand(shape) > 0.3
    if mask.ndim == 2:
        expected = (
            interpolate(mask[:, None].float(), size=new_length, mode="nearest")
            .squeeze(1)
            .bool()
        )
    else:
        expected = interpolate(mask.float(), size=new_length, mode="nearest").bool()
    actual = BaseModule.align_mask(None, mask, new_length)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cuda_prefetch_cpu_fallback_preserves_order_and_tail(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data = TensorDataset(torch.arange(11))
    loader = CudaPrefetchDataLoader(data, batch_size=4, num_workers=0)
    batches = list(loader)
    assert [len(b[0]) for b in batches] == [4, 4, 3]
    torch.testing.assert_close(torch.cat([b[0] for b in batches]), torch.arange(11))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only stream check")
def test_cuda_prefetch_nested_metadata_and_multiple_epochs():
    torch.cuda.set_device(0)
    data = [{"x": torch.full((32, 8), float(i)), "name": str(i)} for i in range(11)]
    loader = CudaPrefetchDataLoader(data, batch_size=4, num_workers=0, pin_memory=True)
    streams = []
    for _ in range(3):
        iterator = iter(loader)
        streams.append(iterator.stream.cuda_stream)
        batches = list(iterator)
        actual = torch.cat([b["x"] for b in batches]).cpu()
        expected = torch.stack([d["x"] for d in data])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert [name for b in batches for name in b["name"]] == list(
            map(str, range(11))
        )
    assert len(set(streams)) == 1
