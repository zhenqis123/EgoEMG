"""Session-v2 fusion keeps centered EMG inside splits and physical segments.

CLI: python -m egoemg.train \
    experiment=fusion/fusion_vits_emgformer_small_session_v2 \
    pretrained_emg_checkpoint=/path/to/emg.ckpt \
    vision_vit_checkpoint=/path/to/vision.ckpt
"""
import json

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from egoemg.datasets.session.fusion import EgoEmgSessionFusionDataset
from egoemg.test_analysis import infer_dataset_type


def make_archive(root, boundary=None, camera_rows=None):
    folder = root / "sample"
    folder.mkdir()

    def store(name, values):
        values = np.asarray(values)
        values.tofile(folder / name)
        return dict(path=name, dtype=values.dtype.str, shape=list(values.shape))

    n = 32
    times = np.arange(n, dtype=np.int64) * 500
    if boundary == "clock_gap":
        times[6:] += 200_000
    elif boundary == "clock_reset":
        times[6:] -= 20_000
    rows = np.array([2, 8, 14, 18, 24, 30] if camera_rows is None else camera_rows)
    emg = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 8)).copy()
    arrays = {
        "emg_filtered": store("emg.dat", emg),
        "timestamps": store("times.dat", times),
        "split_id": store("splits.dat", np.r_[np.zeros(16, np.int8),
                                                 np.full(16, 2, np.int8)]),
        "joint_angles": store("angles.dat", np.broadcast_to(
            np.arange(n, dtype=np.float32)[:, None], (n, 20)).copy()),
        "joint_angles_valid": store("labels.dat", np.ones(n, np.uint8)),
        "wrist_angles": store("wrists.dat", np.full((n, 2), 90, np.float32)),
        "wrist_valid": store("wrist_mask.dat", np.ones(n, np.uint8)),
    }
    arrays["emg_filtered"]["unit"] = "mV"
    if boundary == "emg_valid":
        valid = np.ones(n, np.uint8)
        valid[7] = 0
        arrays[boundary] = store("emg_valid.dat", valid)
    elif boundary in ("segment_id", "recording_index", "interaction_id"):
        arrays[boundary] = store("boundary.dat", np.r_[np.zeros(6, np.int32),
                                                       np.ones(26, np.int32)])
    offsets = np.zeros((len(rows), 2, 2), np.uint64)
    data = bytearray()
    for frame, row in enumerate(rows):
        _, jpeg = cv2.imencode(".jpg", np.full((256, 256, 3), row, np.uint8))
        offsets[frame, :, 0] = len(data)
        offsets[frame, :, 1] = len(jpeg)
        data.extend(jpeg.tobytes())
    (folder / "crops.data").write_bytes(data)
    variant = dict(crop_id="test", generation_complete=True,
                   storage=dict(path="crops.data"), arrays=dict(
                       valid=store("crop_valid.dat", np.ones((len(rows), 2), np.uint8)),
                       offsets=store("offsets.dat", offsets)))
    stream = dict(arrays=dict(timestamps=store("camera_times.dat", times[rows])),
                  hand_crops=dict(active_id="test", variants=[variant]))
    session = dict(session_id="sample", subject="0001", hands=dict(
        left=dict(rows=n, arrays=arrays), right=None), cameras=dict(
            showee_head=dict(streams=dict(camera=stream))))
    (folder / "session.json").write_text(json.dumps(session))
    (root / "dataset.json").write_text(json.dumps(dict(
        units=dict(emg="mV"), sessions=[dict(path="sample/session.json",
                                            source_group="showee")]))
    )
    stats = root / "stats.json"
    stats.write_text(json.dumps({
        "emg_filtered_left": {"mean": [1] * 8, "std": [2] * 8},
    }))
    return stats


def test_fusion_center_alignment_normalization_and_split_boundaries(tmp_path):
    stats = make_archive(tmp_path)
    kwargs = dict(window_length=8, sample_interval_us=0,
                  auto_build_index=True, norm_stats_path=stats)
    train = EgoEmgSessionFusionDataset(tmp_path, ["train"], **kwargs)
    held = EgoEmgSessionFusionDataset(tmp_path, ["gesture"], **kwargs)
    assert len(train) == len(held) == 1
    sample = train[0]
    assert sample["label_row"] == 8 and sample["emg_start_row"] == 4
    assert sample["emg"].shape == (8, 8)
    expected = np.broadcast_to((np.arange(4, 12) - 1) / (2 + 1e-6), (8, 8))
    np.testing.assert_allclose(sample["emg"], expected, rtol=1e-6)
    assert sample["vision_img"].shape == (3, 256, 256)
    assert sample["vision_img"].dtype == np.float32
    assert sample["joint_angles"].shape == (22, 1)
    assert not sample["label_valid_mask"][20:].any()
    assert sample["vision_valid_mask"]
    assert held[0]["label_row"] == 24
    assert train.session_index_groups()[0].tolist() == [0]
    config = OmegaConf.create({"dataset": {"test": [{"_target_":
        "egoemg.datasets.session.fusion.EgoEmgSessionFusionDataset"}]}})
    assert infer_dataset_type(config) == "egoemg_session_v2"


@pytest.mark.parametrize("boundary", ["emg_valid", "segment_id", "recording_index",
                                    "interaction_id", "clock_gap", "clock_reset"])
def test_fusion_rejects_windows_crossing_invalid_spans(tmp_path, boundary):
    stats = make_archive(tmp_path, boundary)
    dataset = EgoEmgSessionFusionDataset(
        tmp_path, ["train"], window_length=8, sample_interval_us=0,
        auto_build_index=True, norm_stats_path=stats,
    )
    assert len(dataset) == 0


def test_fusion_jitter_moves_all_modalities_and_preserves_anchor_count(tmp_path, monkeypatch):
    stats = make_archive(tmp_path, camera_rows=[2, 8, 10, 12, 14, 18, 24, 26, 28, 30])
    kwargs = dict(window_length=8, sample_interval_us=4000,
                  auto_build_index=True, norm_stats_path=stats, return_uint8=True)
    fixed = EgoEmgSessionFusionDataset(tmp_path, ["train"], **kwargs)
    train = EgoEmgSessionFusionDataset(tmp_path, ["train"], jitter=True, **kwargs)
    held = EgoEmgSessionFusionDataset(tmp_path, ["gesture"], **kwargs)
    assert len(train) == len(fixed) == 1
    np.testing.assert_array_equal(train.samples, fixed.samples)
    monkeypatch.setattr(np.random, "randint", lambda lo, hi: hi - 1)
    sample = train[0]
    assert sample["anchor_label_row"] == 8
    assert sample["label_row"] == 12  # Row 14 would cross the split at 16.
    assert sample["emg_start_row"] == 8 and sample["window_offset_samples"] == 4
    assert np.all(sample["joint_angles"][:20] == 12)
    assert np.all(sample["vision_img"] == 12)
    expected = np.broadcast_to((np.arange(8, 16) - 1) / (2 + 1e-6), (8, 8))
    np.testing.assert_allclose(sample["emg"], expected, rtol=1e-6)
    assert held[0]["label_row"] == held[0]["label_row"] == 24
    # Sampling must not mutate anchors used by samplers and diagnostics.
    np.testing.assert_array_equal(train.samples, fixed.samples)


@pytest.mark.parametrize("boundary", ["emg_valid", "segment_id", "recording_index",
                                    "interaction_id", "clock_gap", "clock_reset"])
def test_jitter_respects_physical_boundaries(tmp_path, boundary):
    stats = make_archive(tmp_path, boundary, camera_rows=[2, 8, 10, 12, 14, 24])
    dataset = EgoEmgSessionFusionDataset(
        tmp_path, ["train"], window_length=4, sample_interval_us=4000,
        auto_build_index=True, norm_stats_path=stats, jitter=True,
    )
    assert all("jitter_edges" not in record and "jitter_invalid_prefix" not in record
               for record in dataset.records)
    for i in range(len(dataset)):
        for _ in range(5):
            sample = dataset[i]
            assert sample["emg_start_row"] + 4 <= 16
            if sample["anchor_label_row"] >= 8:
                assert sample["emg_start_row"] >= 6


def test_jitter_rejects_evaluation_split(tmp_path):
    with pytest.raises(ValueError, match="train split"):
        EgoEmgSessionFusionDataset(tmp_path, ["gesture"], norm_mode=None, jitter=True)
