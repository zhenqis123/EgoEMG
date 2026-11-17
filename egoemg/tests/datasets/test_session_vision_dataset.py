"""Synthetic crop/clock/split checks; no video, MANO or pretrained weights."""

import json
import os

import cv2
import numpy as np
import pytest

from egoemg.datasets.session.crops import (EgoEmgSessionVisionDataset)
from egoemg.datasets.session.alignment import (align_camera_rows)


def test_align_camera_rows_handles_clock_reset_and_stale_samples():
    rows, valid, ages = align_camera_rows(
        np.array([100_000, 101_000, 1000, 2000]),
        np.array([0, 1500, 60_000, 100_500]),
    )
    assert rows.tolist() == [2, 2, 3, 0]
    assert valid.tolist() == [False, True, False, True]
    assert ages[-1] == 500


@pytest.fixture
def crop_dataset(tmp_path):
    folder = tmp_path / "sessions" / "sample"
    folder.mkdir(parents=True)

    def store(name, data):
        data = np.asarray(data)
        data.tofile(folder / name)
        return dict(path=name, dtype=data.dtype.str, shape=list(data.shape))

    arrays = {
        "timestamps": store("times.dat", np.array([1000, 2000, 3000], np.int64)),
        "split_id": store("split.dat", np.array([0, 2, 3], np.int8)),
        "joint_angles_valid": store("valid.dat", np.ones(3, np.uint8)),
        "joint_angles": store("angles.dat", np.ones((3, 20), np.float32)),
        "wrist_angles": store("wrist.dat", np.full((3, 2), 90, np.float32)),
        "wrist_valid": store("wrist_valid.dat", np.array([1, 0, 1], np.uint8)),
    }
    _, jpeg = cv2.imencode(".jpg", np.full((256, 256, 3), 127, np.uint8))
    (folder / "images.data").write_bytes(jpeg.tobytes())
    offsets = np.zeros((3, 2, 2), np.uint64)
    offsets[:, :, 1] = len(jpeg)
    variant = dict(
        crop_id="test",
        generation_complete=True,
        storage=dict(path="images.data"),
        arrays={
            "offsets": store("offsets.dat", offsets),
            "valid": store("crop_valid.dat", np.ones((3, 2), np.uint8)),
        },
    )
    stream = dict(
        arrays={"timestamps": arrays["timestamps"]},
        hand_crops=dict(active_id="test", variants=[variant]),
    )
    session = dict(
        session_id="sample",
        hands={"left": dict(arrays=arrays), "right": None},
        cameras={"showee_head": dict(streams={"camera": stream})},
    )
    (folder / "session.json").write_text(json.dumps(session))
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            dict(
                sessions=[
                    dict(path="sessions/sample/session.json", source_group="recent")
                ]
            )
        )
    )
    return tmp_path, folder


def test_split_mask_rgb_and_stale_index(crop_dataset):
    root, folder = crop_dataset
    with pytest.raises(RuntimeError, match="Missing/stale"):
        EgoEmgSessionVisionDataset(root, ["gesture"])
    dataset = EgoEmgSessionVisionDataset(root, ["gesture"], auto_build_index=True)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["camera_frame"] == sample["label_row"] == 1
    assert sample["joint_angles"].shape == (22, 1)
    assert not sample["label_valid_mask"][20:].any()
    np.testing.assert_allclose(sample["joint_angles"][20:], np.pi / 2)
    raw = EgoEmgSessionVisionDataset(root, ["gesture"], return_uint8=True)[0]
    assert raw["vision_img"].dtype == np.uint8
    np.testing.assert_allclose(
        sample["vision_img"],
        (raw["vision_img"].astype(np.float32) / 255 - dataset.mean) / dataset.std,
    )
    split_path = folder / "split.dat"
    old = split_path.stat().st_mtime_ns
    np.array([0, 0, 3], np.int8).tofile(split_path)
    os.utime(split_path, ns=(old + 1_000_000, old + 1_000_000))
    with pytest.raises(RuntimeError, match="Missing/stale"):
        EgoEmgSessionVisionDataset(root, ["gesture"])
    assert (
        len(EgoEmgSessionVisionDataset(root, ["gesture"], auto_build_index=True)) == 0
    )


def test_native_rgb_decode_and_legacy_fallback_preserve_colors(crop_dataset, monkeypatch):
    root, folder = crop_dataset
    image = np.zeros((256, 256, 3), np.uint8)
    image[:, :, 0] = np.arange(256, dtype=np.uint8)[None, :]
    image[:, :, 1] = 63
    image[:, :, 2] = np.arange(256, dtype=np.uint8)[:, None]
    _, jpeg = cv2.imencode(".jpg", image)
    (folder / "images.data").write_bytes(jpeg.tobytes())
    offsets = np.zeros((3, 2, 2), np.uint64)
    offsets[:, :, 1] = len(jpeg)
    offsets.tofile(folder / "offsets.dat")
    expected = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)[:, :, ::-1].transpose(2, 0, 1)
    dataset = EgoEmgSessionVisionDataset(
        root, ["train"], return_uint8=True, auto_build_index=True,
    )
    np.testing.assert_array_equal(dataset[0]["vision_img"], expected)
    monkeypatch.delattr(cv2, "IMREAD_COLOR_RGB", raising=False)
    np.testing.assert_array_equal(dataset[0]["vision_img"], expected)
