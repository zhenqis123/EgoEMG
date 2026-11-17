"""Protected initialization, missing-hand fallback, gradients and paired RNG."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from egoemg.training.datamodule import WindowedEmgDataModule
from egoemg.datasets.session.alignment import (dependency_signature)
from egoemg.datasets.session.fusion import (
    EgoEmgSessionFullFrameFusionDataset,
)
from egoemg.training.prediction import EmgPredictionModule
from egoemg.models.modules.bimanual_fusion import BimanualResidualFusion, load_branch


def test_checkpoint_restoration_is_strict(tmp_path):
    from omegaconf import OmegaConf

    original = Vision()
    state = {"model." + key: value for key, value in original.state_dict().items()}
    checkpoint = dict(
        state_dict=state,
        hyper_parameters=dict(
            module_conf=OmegaConf.create(
                {
                    "_target_": "egoemg.tests.models.test_bimanual_fusion.Vision",
                }
            )
        ),
    )
    path = tmp_path / "branch.ckpt"
    torch.save(checkpoint, path)
    restored = load_branch(path)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    state.pop("model.head.0.weight")
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_branch(path)


class Vision(nn.Module):
    bimanual_output = True

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 384)
        )
        self.head = nn.Sequential(nn.Linear(384, 44))
        self.register_buffer("image_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("image_std", torch.ones(1, 3, 1, 1))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_proj = nn.Linear(256, 256)

    def forward(self, x):
        return self.output_proj(x.transpose(1, 2)).transpose(1, 2)


class Emg(nn.Module):
    def __init__(self):
        super().__init__()
        self.featurizer = nn.Sequential(nn.Conv1d(8, 256, 1), nn.BatchNorm1d(256))
        self.decoder = Decoder()


def test_frozen_identity_and_missing_emg_after_learning(monkeypatch):
    vision, emg = Vision(), Emg()
    monkeypatch.setattr(
        "egoemg.models.modules.bimanual_fusion.load_branch",
        lambda path: vision if path == "vision" else emg,
    )
    model = BimanualResidualFusion("vision", "emg").train()
    assert not vision.training and not emg.training
    assert model.head.training
    batch = dict(
        vision_img=torch.rand(2, 3, 16, 16),
        emg=torch.randn(2, 2, 8, 20),
        emg_valid_mask=torch.tensor([[True, False], [True, True]]),
        joint_angles=torch.randn(2, 44, 1),
        label_valid_mask=torch.ones(2, 44, 1, dtype=torch.bool),
    )
    frozen = {
        k: v.clone() for k, v in model.state_dict().items() if not k.startswith("head.")
    }
    with torch.no_grad():
        baseline = vision.head(vision.backbone(batch["vision_img"])).reshape(4, 22, 1)
    pred, targets, mask = model(batch)
    torch.testing.assert_close(pred, baseline, rtol=0, atol=0)
    loss = (pred - targets).abs().mean() + 0.1 * model._last_delta_l1
    loss.backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in vision.parameters())
    assert all(p.grad is None for p in emg.parameters())
    with torch.no_grad():
        model.head[-1].bias.fill_(1)
    pred, _, _ = model(batch)
    torch.testing.assert_close(pred[1], baseline[1], rtol=0, atol=0)
    assert model._last_delta_l1.item() == 1
    model.force_zero_correction = True
    torch.testing.assert_close(model(batch)[0], baseline, rtol=0, atol=0)
    for key, value in frozen.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "freeze_emg,freeze_vision", [(False, True), (True, False), (False, False)]
)
def test_unfrozen_branches_receive_gradients_and_update(
    monkeypatch, freeze_emg, freeze_vision
):
    vision, emg = Vision(), Emg()
    emg.head = nn.Linear(256, 22)  # diagnostic-only head must stay frozen.
    monkeypatch.setattr(
        "egoemg.models.modules.bimanual_fusion.load_branch",
        lambda path: vision if path == "vision" else emg,
    )
    model = BimanualResidualFusion(
        "vision", "emg", freeze_emg=freeze_emg, freeze_vision=freeze_vision
    ).train()
    assert vision.training == (not freeze_vision)
    assert emg.training == (not freeze_emg)
    assert not any(p.requires_grad for p in emg.head.parameters())
    # After the zero-initialized last layer learns, gradients must reach the
    # selected encoders, rather than merely having requires_grad=True.
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.02)
    batch = dict(
        vision_img=torch.rand(2, 3, 16, 16),
        emg=torch.randn(2, 2, 8, 20),
        emg_valid_mask=torch.ones(2, 2, dtype=torch.bool),
        joint_angles=torch.randn(2, 44, 1),
        label_valid_mask=torch.ones(2, 44, 1, dtype=torch.bool),
    )
    snapshots = {name: value.clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5
    )
    prediction, target, _ = model(batch)
    (prediction - target).square().mean().backward()
    for branch, frozen in [
        (vision, freeze_vision),
        (emg.featurizer, freeze_emg),
        (emg.decoder, freeze_emg),
    ]:
        grads = [p.grad for p in branch.parameters()]
        assert (
            all(g is None for g in grads)
            if frozen
            else any(g is not None and g.abs().sum() > 0 for g in grads)
        )
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, snapshots[name], rtol=0, atol=0)
    model.eval()
    assert not model.vision.training and not model.emg_model.training
    model.train()
    batch["emg_valid_mask"].zero_()
    before_mean = emg.featurizer[1].running_mean.clone()
    model(batch)
    torch.testing.assert_close(
        emg.featurizer[1].running_mean, before_mean, rtol=0, atol=0
    )




def test_batch_augmentation_flattens_hands_once_without_touching_targets():
    calls = []

    class Aug:
        mixup_prob = 0

        def __call__(self, batch):
            calls.append(batch["emg"].shape)
            return {"emg": batch["emg"] + torch.arange(4)[:, None, None]}

    fake = SimpleNamespace(
        model=SimpleNamespace(bimanual_output=True),
        batch_aug=Aug(),
        trainer=SimpleNamespace(training=True),
    )
    batch = dict(emg=torch.zeros(2, 2, 8, 20), joint_angles=torch.randn(2, 44, 1))
    targets = batch["joint_angles"].clone()
    result = EmgPredictionModule.on_after_batch_transfer(fake, batch, 0)
    assert calls == [torch.Size([4, 8, 20])]
    assert result["emg"][:, :, 0, 0].tolist() == [[0, 1], [2, 3]]
    torch.testing.assert_close(result["joint_angles"], targets)
    fake.trainer.training = False
    EmgPredictionModule.on_after_batch_transfer(fake, batch, 0)
    assert len(calls) == 1


def test_mismatch_controls_preserve_condition_and_report_uncovered_rows():
    from scripts.analysis.evaluate_cached_emg_time_mismatch import mismatch_indices

    session = ["a", "a", "a", "b"]
    objects = np.array([[0, 8], [0, 8], [8, 8], [0, 8]])
    valid = np.ones((4, 2), bool)
    times = np.array([0, 5_000_000, 10_000_000, 20_000_000])
    indices = mismatch_indices(session, objects, times, valid)
    assert indices[0, 0] == 1 and indices[1, 0] == 0
    assert indices[2, 0] == -1 and (indices[3] == -1).all()
    for row, hand in np.argwhere(indices >= 0):
        other = indices[row, hand]
        assert session[row] == session[other]
        assert objects[row, hand] == objects[other, hand]
        assert abs(times[row] - times[other]) > 4_000_000


@pytest.fixture
def paired_root(tmp_path):
    folder = tmp_path / "sessions/sample"
    folder.mkdir(parents=True)

    def store(name, values):
        values = np.asarray(values)
        values.tofile(folder / name)
        return dict(
            path=name, shape=list(values.shape), dtype=str(values.dtype), unit="mV"
        )

    anchors = np.array([6, 10, 18, 24, 30])
    time = np.arange(40, dtype=np.int64) * 500
    hands = {}
    split_columns = []
    for index, side in enumerate(("left", "right")):
        split = np.where(np.arange(40) >= 20, 4 if index == 0 else 0, 0).astype(np.int8)
        valid = np.ones(40, np.uint8)
        valid[12] = 0
        hands[side] = dict(
            arrays=dict(
                timestamps=store(side + "time", time),
                split_id=store(side + "split", split),
                emg_filtered=store(
                    side + "emg", np.arange(320, dtype=np.float32).reshape(40, 8)
                ),
                emg_valid=store(side + "valid", valid),
                joint_angles=store(side + "angles", np.ones((40, 20), np.float32)),
                joint_angles_valid=store(side + "labels", np.ones(40, np.uint8)),
                wrist_angles=store(side + "wrist", np.zeros((40, 2), np.float32)),
                wrist_valid=store(side + "wv", np.ones(40, np.uint8)),
            )
        )
        split_columns.append(split[anchors])
    (folder / "video.mp4").touch()
    session = dict(
        session_id="sample",
        source_group="recent",
        hands=hands,
        cameras=dict(
            showee_head=dict(
                streams=dict(
                    camera=dict(
                        arrays=dict(timestamps=store("camera_time", time[anchors])),
                        video=dict(path="video.mp4"),
                    )
                )
            )
        ),
    )
    path = folder / "session.json"
    path.write_text(json.dumps(session))
    (tmp_path / "dataset.json").write_text(
        json.dumps(dict(sessions=[dict(path="sessions/sample/session.json")]))
    )
    cache = tmp_path / "fullframe_256_index"
    cache.mkdir()
    np.save(cache / "sample.npy", np.zeros((5, 256, 256, 3), np.uint8))
    np.savez(
        cache / "sample.npz",
        frames=np.arange(5),
        rows=np.column_stack((anchors, anchors)),
        splits=np.column_stack(split_columns),
        times=time[anchors],
    )
    (cache / "sample.json").write_text(
        json.dumps(dict(signature=dependency_signature(path, session, 400000)))
    )
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {
                f"emg_filtered_{s}": dict(mean=[0] * 8, std=[1] * 8)
                for s in ("left", "right")
            }
        )
    )
    return tmp_path, stats


def test_windows_jitter_exclusions_and_same_vision_pool(paired_root):
    root, stats = paired_root
    train = EgoEmgSessionFullFrameFusionDataset(
        root, ["train"], window_length=8, norm_stats_path=stats, jitter=True
    )
    assert len(train) == 5  # retain RGB even when a hand has no legal EMG.
    offsets = [train[0]["window_offset_samples"].copy() for _ in range(40)]
    assert len(np.unique(np.array(offsets)[:, 0])) > 1
    assert train[1]["emg_valid_mask"].tolist() == [False, False]  # invalid EMG at row12
    assert train[2]["emg_valid_mask"].tolist() == [False, True]  # left split boundary
    assert not train[4]["emg_valid_mask"][0]  # held-out hand cannot enter train.
    sample = train[0]
    for side in range(2):
        start = sample["emg_start_row"][side]
        assert 0 <= start <= 4 and start + 8 <= 12
    val = EgoEmgSessionFullFrameFusionDataset(
        root, ["recent_object"], window_length=8, norm_stats_path=stats, jitter=False
    )
    assert len(val) == 2
    assert val[0]["emg_valid_mask"].tolist() == [True, False]
    np.testing.assert_array_equal(val[0]["emg"], val[0]["emg"])
    assert not val[0]["window_offset_samples"].any()
    from egoemg.datasets.session_batch import SessionBatchDataset

    wrapped = SessionBatchDataset(val, WindowedEmgDataModule._collate_fn)
    batch = wrapped.__getitems__([0, 1])
    assert batch["emg"].shape == (2, 2, 8, 8)
    torch.testing.assert_close(batch["emg"][0], torch.from_numpy(val[0]["emg"]))
    with pytest.raises(ValueError, match="training-only"):
        EgoEmgSessionFullFrameFusionDataset(
            root, ["recent_object"], norm_stats_path=stats, jitter=True
        )


def test_short_recording_falls_back_to_vision(paired_root):
    root, stats = paired_root
    dataset = EgoEmgSessionFullFrameFusionDataset(
        root,
        ["train"],
        window_length=80,
        norm_stats_path=stats,
    )
    assert len(dataset) == 5
    assert not dataset[0]["emg_valid_mask"].any()
    assert not dataset[0]["emg"].any()


