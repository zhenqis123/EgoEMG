"""Independent hand masks, physical camera mapping and ordered 44-angle output."""

import json

import numpy as np
import torch

from egoemg.datasets.session.fullframe import (EgoEmgSessionFullFrameDataset)
from egoemg.datasets.session.alignment import (build_alignment, dependency_signature)
from egoemg.models.modules.bimanual_vision import (
    BimanualVisionViTPose,
    expand_bimanual_metadata,
)


def test_fullframe_masks_other_hand_split_without_crop_dependency(tmp_path):
    folder = tmp_path / "sessions/sample"
    folder.mkdir(parents=True)

    def store(name, values):
        values = np.asarray(values)
        values.tofile(folder / name)
        return dict(path=name, shape=list(values.shape), dtype=str(values.dtype))

    times = store("times.dat", np.array([1000, 501000], np.int64))
    hands = {}
    for index, side in enumerate(("left", "right")):
        hands[side] = dict(arrays=dict(
            timestamps=times,
            split_id=store(side + "split.dat", np.full(2, index * 4, np.int8)),
            joint_angles_valid=store(side + "valid.dat", np.ones(2, np.uint8)),
            joint_angles=store(side + "angles.dat", np.full((2, 20), index + 1, np.float32)),
            wrist_angles=store(side + "wrist.dat", np.full((2, 2), 90, np.float32)),
            wrist_valid=store(side + "wv.dat", np.array([1, 0], np.uint8)),
        ))
    (folder / "video.mp4").touch()
    session = dict(
        session_id="sample", source_group="recent", hands=hands,
        cameras=dict(showee_head=dict(streams=dict(camera=dict(
            arrays=dict(timestamps=times), video=dict(
                path="video.mp4",
                frame_index_map=store("map.dat", np.array([2, 5], np.int64)),
            ),
        )))),
    )
    path = folder / "session.json"
    path.write_text(json.dumps(session))
    (tmp_path / "dataset.json").write_text(json.dumps(dict(
        sessions=[dict(path="sessions/sample/session.json")],
    )))
    alignment = build_alignment(folder, session)
    assert alignment["physical"].tolist() == [2, 5]
    assert alignment["rows"].tolist() == [[0, 0], [1, 1]]
    cache = tmp_path / "fullframe_256_index"
    cache.mkdir()
    np.save(cache / "sample.npy", np.full((2, 256, 256, 3), 127, np.uint8))
    np.savez(cache / "sample.npz", **alignment)
    (cache / "sample.json").write_text(json.dumps(dict(
        signature=dependency_signature(path, session, 400_000),
    )))
    train = EgoEmgSessionFullFrameDataset(tmp_path, ["train"])
    val = EgoEmgSessionFullFrameDataset(tmp_path, ["recent_object"])
    assert len(train) == len(val) == 2
    assert train[0]["vision_img"].shape == (3, 256, 256)
    assert train[0]["label_valid_mask"][:22].all()
    assert not train[0]["label_valid_mask"][22:].any()
    assert not val[0]["label_valid_mask"][:22].any()
    assert val[0]["label_valid_mask"][22:].all()
    assert not val[1]["label_valid_mask"][42:].any()
    np.testing.assert_allclose(val[0]["joint_angles"][42:], np.pi / 2)


def test_two_heads_are_independent_hands_not_temporal_derivatives(monkeypatch):
    import timm
    from torch import nn

    class FakeBackbone(nn.Module):
        def forward(self, image):
            return image.mean((2, 3)).mean(1, keepdim=True).expand(-1, 384)

    monkeypatch.setattr(timm, "create_model", lambda *a, **kw: FakeBackbone())
    model = BimanualVisionViTPose(pretrained=False, backbone_type="vit_small")
    batch = dict(
        vision_img=torch.zeros(2, 3, 256, 256, dtype=torch.uint8),
        joint_angles=torch.arange(88).reshape(2, 44, 1).float(),
        label_valid_mask=torch.ones(2, 44, 1, dtype=torch.bool),
        emg=torch.zeros(2, 8, 1), split_id=torch.tensor([[0, 4], [2, 3]]),
        episode_id=["a", "b"], dataset_name=["egoemg", "showee"],
    )
    predictions, targets, mask = model(batch)
    assert predictions.shape == targets.shape == mask.shape == (4, 22, 1)
    torch.testing.assert_close(targets[:, 0, 0], torch.tensor([0., 22., 44., 66.]))
    metadata = expand_bimanual_metadata(batch)
    assert metadata["split_id"].tolist() == [0, 4, 2, 3]
    assert metadata["episode_id"] == ["a", "a", "b", "b"]
    assert metadata["dataset_name"] == ["egoemg", "egoemg", "showee", "showee"]
    predictions.sum().backward()
    assert model.head[-1].weight.grad is not None
