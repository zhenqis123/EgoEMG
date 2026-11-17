"""Small session-v2 fixtures for split, continuity and label semantics.

CLI smoke: python -m egoemg.train \
    experiment=emgformer/regression_egoemg_session_v2 --cfg job
"""

import json

import numpy as np
import torch

from egoemg.datasets.session.emg import (EgoEmgSessionDataset)
from egoemg.models.modules.base import BaseModule
from scripts.data.session.compute_session_emg_stats import compute


def test_session_windows_and_channel_validity(tmp_path):
    root = tmp_path
    folder = root / "sessions" / "sample"
    (folder / "left").mkdir(parents=True)
    n = 16
    arrays = {}

    def add(name, values):
        path = f"left/{name}.dat"
        values.tofile(folder / path)
        arrays[name] = dict(
            path=path, dtype=str(values.dtype), shape=list(values.shape)
        )

    emg = np.arange(n * 8, dtype=np.float32).reshape(n, 8) * 0.001
    add("emg_filtered", emg)
    arrays["emg_filtered"]["unit"] = "mV"
    joints = np.ones((n, 20), np.float32)
    joints[13, 0] = np.nan
    add("joint_angles", joints)
    add("wrist_angles", np.full((n, 2), 180, np.float32))
    wrist_valid = np.ones(n, np.uint8)
    wrist_valid[12] = 0
    add("wrist_valid", wrist_valid)
    label_valid = np.ones(n, np.uint8)
    label_valid[8:12] = 0
    add("joint_angles_valid", label_valid)
    split = np.r_[np.zeros(8, np.int8), np.full(8, 2, np.int8)]
    add("split_id", split)
    add("timestamps", np.arange(n, dtype=np.int64) * 500)
    segment = np.r_[np.zeros(6, np.int32), np.ones(10, np.int32)]
    add("segment_id", segment)
    emg_valid = np.ones(n, np.uint8)
    emg_valid[4] = 0
    add("emg_valid", emg_valid)
    session = dict(
        session_id="sample", subject="0001", hands={
            "left": dict(rows=n, arrays=arrays),
            "right": dict(rows=n, arrays=arrays),
        },
    )
    (folder / "session.json").write_text(json.dumps(session))
    (root / "dataset.json").write_text(json.dumps(dict(
        units=dict(emg="mV"),
        sessions=[dict(path="sessions/sample/session.json", source_group="recent")],
    )))

    train = EgoEmgSessionDataset(root, "left", ["train"], 4, 2)
    held_out = EgoEmgSessionDataset(root, "left", ["gesture"], 4, 2)
    assert train._starts.tolist() == [0]
    assert held_out._starts.tolist() == [10, 12]
    sample = held_out[1]
    assert sample["emg"].shape == (8, 4)
    assert sample["joint_angles"].shape == (22, 4)
    assert sample["label_valid_mask"].shape == (22, 4)
    assert not sample["label_valid_mask"][20:, 0].any()
    assert not sample["label_valid_mask"][0, 1]
    assert sample["joint_angles"][0, 1] == 0
    np.testing.assert_allclose(sample["joint_angles"][20:, 1], np.pi)
    assert sample["held_out_stage"] and not sample["held_out_user"]
    assert sample["episode_id"] == "sample"
    assert len(held_out.session_index_groups()[0]) == len(held_out)

    stats = compute(root, root / "stats.json")
    # The held-out gesture's much larger samples must not enter the estimate.
    np.testing.assert_allclose(
        stats["emg_filtered_left"]["mean"],
        emg[np.r_[0:4, 5:8]].mean(axis=0), rtol=1e-5,
    )


def test_align_mask_preserves_wrist_channels():
    mask = torch.ones(1, 22, 4, dtype=torch.bool)
    mask[:, 20:, :] = False
    aligned = BaseModule.align_mask(None, mask, 2)
    assert aligned.shape == (1, 22, 2)
    assert not aligned[:, 20:, :].any()
    assert aligned[:, :20, :].all()


def test_session_jitter_shifts_targets_and_stays_inside_block(tmp_path, monkeypatch):
    # Reuse a native archive: train ends at row 16, with row-valued EMG/labels.
    from egoemg.tests.datasets.test_session_fusion_dataset import make_archive

    make_archive(tmp_path)
    train = EgoEmgSessionDataset(tmp_path, "left", ["train"], 4, 5, jitter=True)
    fixed = EgoEmgSessionDataset(tmp_path, "left", ["train"], 4, 5)
    np.testing.assert_array_equal(train._starts, fixed._starts)
    assert train._starts.tolist() == [0, 5, 10]
    monkeypatch.setattr(np.random, "randint", lambda high: high - 1)
    for index, expected in enumerate([4, 9, 12]):
        sample = train[index]
        assert sample["emg_start_row"] == expected
        np.testing.assert_allclose(sample["emg"][0], np.arange(expected, expected + 4), rtol=2e-6)
        np.testing.assert_array_equal(sample["joint_angles"][0], np.arange(expected, expected + 4))
    assert fixed[0]["emg_start_row"] == fixed[0]["emg_start_row"] == 0


def test_session_jitter_never_loses_all_supervision(tmp_path, monkeypatch):
    import json
    from egoemg.tests.datasets.test_session_fusion_dataset import make_archive

    make_archive(tmp_path)
    folder = tmp_path / "sample"
    session = json.loads((folder / "session.json").read_text())
    spec = session["hands"]["left"]["arrays"]["joint_angles_valid"]
    labels = np.memmap(folder / spec["path"], dtype=spec["dtype"], mode="r+", shape=(32,))
    labels[:] = 0
    labels[0] = 1
    labels.flush()
    train = EgoEmgSessionDataset(tmp_path, "left", ["train"], 4, 5, jitter=True)
    monkeypatch.setattr(np.random, "randint", lambda high: high - 1)
    assert train[0]["emg_start_row"] == 0
    assert train[0]["label_valid_mask"].any()


def test_session_jitter_is_train_only(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="train split"):
        EgoEmgSessionDataset(tmp_path, "left", ["gesture"], 4, 2, jitter=True)


def test_jitter_worker_streams_are_random_and_seed_reproducible(tmp_path):
    from torch.utils.data import DataLoader, Subset
    from egoemg.tests.datasets.test_session_fusion_dataset import make_archive

    make_archive(tmp_path)
    dataset = EgoEmgSessionDataset(tmp_path, "left", ["train"], 4, 5, jitter=True)

    def offsets():
        loader = DataLoader(
            Subset(dataset, [0] * 24), batch_size=1, num_workers=2,
            generator=torch.Generator().manual_seed(52),
        )
        return [int(batch["window_offset_samples"].item()) for batch in loader]

    first, second = offsets(), offsets()
    assert first == second
    assert len(set(first[::2])) > 1 and len(set(first[1::2])) > 1
    assert first[::2] != first[1::2]


def test_session_jitter_caps_offsets_at_physical_boundaries(tmp_path, monkeypatch):
    from egoemg.tests.datasets.test_session_fusion_dataset import make_archive

    monkeypatch.setattr(np.random, "randint", lambda high: high - 1)
    for boundary in ("emg_valid", "segment_id", "recording_index",
                     "interaction_id", "clock_gap", "clock_reset"):
        root = tmp_path / boundary
        root.mkdir()
        make_archive(root, boundary)
        dataset = EgoEmgSessionDataset(root, "left", ["train"], 4, 5, jitter=True)
        for i in range(len(dataset)):
            start = dataset[i]["emg_start_row"]
            end = start + 4
            assert end <= 16
            if boundary == "emg_valid":
                assert not start <= 7 < end
            else:
                assert not start < 6 < end


def test_direct_target_buffer_preserves_nonfinite_masks_and_readonly_inputs(tmp_path):
    from egoemg.tests.datasets.test_session_fusion_dataset import make_archive

    make_archive(tmp_path)
    folder = tmp_path / "sample"
    joints = np.memmap(folder / "angles.dat", np.float32, mode="r+", shape=(32, 20))
    joints[4, 0], joints[5, 1], joints[6, 2] = np.inf, -np.inf, np.nan
    joints.flush()
    wrists = np.memmap(folder / "wrists.dat", np.float32, mode="r+", shape=(32, 2))
    wrists[1, 0] = np.nan
    wrists.flush()
    labels = np.memmap(folder / "labels.dat", np.uint8, mode="r+", shape=(32,))
    labels[2] = 0
    labels.flush()
    before = {name: (folder / name).read_bytes()
              for name in ("emg.dat", "angles.dat", "wrists.dat", "labels.dat")}
    dataset = EgoEmgSessionDataset(tmp_path, "left", ["train"], 8, 2)
    sample = dataset[0]
    target, mask = sample["joint_angles"], sample["label_valid_mask"]
    assert target.flags.c_contiguous and mask.flags.c_contiguous
    assert np.isfinite(target).all()
    assert not mask[:, 2].any()
    assert np.all(target[:20, 2] == 2)  # Masked finite values remain intact.
    for channel, row in ((0, 4), (1, 5), (2, 6), (20, 1)):
        assert target[channel, row] == 0 and not mask[channel, row]
    assert all((folder / name).read_bytes() == data for name, data in before.items())
