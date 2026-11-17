"""Batched shared-buffer reads must preserve item order, labels and RNG draws."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader

from egoemg.training.datamodule import WindowedEmgDataModule
from egoemg.datasets.session.emg import (EgoEmgSessionDataset)
from egoemg.datasets.session.fusion import EgoEmgSessionFusionDataset
from egoemg.datasets.session.crops import (EgoEmgSessionVisionDataset)
from egoemg.datasets.session_batch import SessionBatchDataset, batch_signature
from egoemg.tests.datasets.test_session_fusion_dataset import make_archive
from egoemg.augmentation.transforms import Compose, ExtractField, RotationAugmentation, ToFloatTensor


@pytest.mark.parametrize("mode", ["emg", "vision", "fusion"])
@pytest.mark.parametrize("workers", [0, 2])
def test_batch_reads_match_item_collation_and_random_streams(tmp_path, mode, workers):
    stats = make_archive(tmp_path, camera_rows=[2, 8, 10, 12, 14, 18, 24, 26, 28, 30])
    joints = np.memmap(
        tmp_path / "sample/angles.dat", np.float32, mode="r+", shape=(32, 20)
    )
    joints[4, 0] = np.nan
    joints.flush()
    transform = Compose([ExtractField("emg"), RotationAugmentation(), ToFloatTensor()])
    if mode == "emg":
        dataset = ConcatDataset(
            [
                EgoEmgSessionDataset(
                    tmp_path, hand, ["train"], 4, 5, jitter=True, transform=transform
                )
                for hand in ["left", "left"]
            ]
        )
    else:
        kwargs = dict(sample_interval_us=4000, auto_build_index=True, return_uint8=True)
        if mode == "vision":
            dataset = EgoEmgSessionVisionDataset(tmp_path, ["train"], **kwargs)
        else:
            dataset = EgoEmgSessionFusionDataset(
                tmp_path,
                ["train"],
                window_length=8,
                norm_stats_path=stats,
                jitter=True,
                transform=transform,
                **kwargs,
            )
    indices = [len(dataset) - 1, 0, 0, len(dataset) - 1] * 3
    if mode == "emg":
        indices[1] = -1  # ConcatDataset resolves this to its last child.
    collate = WindowedEmgDataModule._collate_fn

    def batches(fast):
        np.random.seed(48)
        torch.manual_seed(48)
        source = SessionBatchDataset(dataset, collate) if fast else dataset
        loader = DataLoader(
            source,
            batch_size=4,
            sampler=indices,
            num_workers=workers,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(48),
        )
        return list(loader)

    reference, candidate = batches(False), batches(True)
    for old, new in zip(reference, candidate):
        assert old.keys() == new.keys()
        for key in old:
            if isinstance(old[key], torch.Tensor):
                assert old[key].dtype == new[key].dtype
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
            else:
                assert old[key] == new[key], key
        for key in ("emg", "joint_angles", "label_valid_mask"):
            assert new[key].is_contiguous()
            if workers:
                assert new[key].is_shared()
    # Readers still support ordinary default-collate DataLoaders.
    assert next(iter(DataLoader(dataset, batch_size=1)))["emg"].shape[0] == 1
    assert batch_signature(dataset) is not None


def test_incompatible_concat_is_not_batched(tmp_path):
    make_archive(tmp_path)
    a = EgoEmgSessionDataset(tmp_path, "left", ["train"], 4, 2)
    b = EgoEmgSessionDataset(tmp_path, "left", ["train"], 8, 2)
    assert batch_signature(ConcatDataset([a, b])) is None


def test_batch_reader_spawn_persistent_workers_and_partial_batch(tmp_path):
    make_archive(tmp_path)
    dataset = EgoEmgSessionDataset(tmp_path, "left", ["train"], 4, 5)
    collate = WindowedEmgDataModule._collate_fn
    source = SessionBatchDataset(dataset, collate)
    loader = DataLoader(
        source,
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        persistent_workers=True,
        collate_fn=collate,
    )
    iterator = iter(loader)
    first, tail = next(iterator), next(iterator)
    assert first["emg"].shape == (2, 8, 4)
    assert tail["emg"].shape == (1, 8, 4)
    assert first["emg"].is_shared()
    assert tail["emg_start_row"].tolist() == [10]
    assert next(iter(loader))["emg_start_row"].tolist() == [0, 5]
    iterator._shutdown_workers()
    assert source.__getitems__([]) == {}


def test_data_module_batch_wrapper_preserves_session_sampler(tmp_path):
    from egoemg.training.datamodule import SessionShuffleSampler

    make_archive(tmp_path)
    dataset = EgoEmgSessionDataset(tmp_path, "left", ["train"], 4, 5)
    module = WindowedEmgDataModule(
        window_length=4,
        stride=5,
        padding=(0, 0),
        batch_size=2,
        num_workers=0,
        batch_fetch=True,
        pin_memory=False,
    )
    loader = module._make_loader(dataset, shuffle=True)
    assert isinstance(loader.dataset, SessionBatchDataset)
    assert isinstance(loader.sampler, SessionShuffleSampler)
    np.testing.assert_array_equal(
        loader.sampler.session_groups[0], dataset.session_index_groups()[0]
    )
    assert next(iter(loader))["joint_angles"].shape == (2, 22, 4)


@pytest.mark.parametrize("mode", ["vision", "fusion"])
def test_float_image_batch_matches_normal_reader(tmp_path, mode):
    stats = make_archive(tmp_path)
    kwargs = dict(sample_interval_us=0, auto_build_index=True, return_uint8=False)
    if mode == "vision":
        dataset = EgoEmgSessionVisionDataset(tmp_path, ["train"], **kwargs)
    else:
        dataset = EgoEmgSessionFusionDataset(
            tmp_path,
            ["train"],
            window_length=8,
            norm_stats_path=stats,
            **kwargs,
        )
    collate = WindowedEmgDataModule._collate_fn
    expected = collate([dataset[0], dataset[0]])
    actual = SessionBatchDataset(dataset, collate).__getitems__([0, 0])
    for key in ("vision_img", "emg", "joint_angles", "label_valid_mask"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert dataset[0]["vision_img"].flags.c_contiguous
