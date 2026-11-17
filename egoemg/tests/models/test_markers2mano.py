"""input ordering, chirality, invalid frames and conversion-only CLI."""

import numpy as np
import pytest
import torch
from torch import nn

from egoemg.markers2mano import MARKER_ORDER, convert_markers, load_model
from egoemg.markers2mano.inference import MODEL_CONFIG


class MarkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.observed = []

    def forward(self, markers):
        self.observed.append(markers.clone())
        rotation = markers.new_tensor([1, 0, 0, 0, 1, 0]).repeat(len(markers), 16)
        beta = torch.arange(10, device=markers.device).float()[None].repeat(len(markers), 1)
        return rotation, beta


def marker_frames(count=3):
    points = np.random.default_rng(7).normal(0, 0.01, (21, 3)).astype(np.float32)
    points[0] = 0
    points[5] = [0.03, 0, 0]
    points[17] = [0.01, 0.02, 0]
    shifts = np.arange(count, dtype=np.float32)[:, None, None] * 0.1
    return points[None] + shifts


@pytest.mark.parametrize("hand", ["left", "right"])
def test_all_markers_and_world_to_canonical_transform(hand):
    model = MarkerModel()
    markers = marker_frames()
    result = convert_markers(model, markers, handedness=hand, batch_size=2)
    observed = torch.cat(model.observed).numpy()
    assert observed.shape == (3, 21, 3)
    assert len(MARKER_ORDER) == 21
    transform = result["world_to_canonical"]
    reconstructed = np.einsum("bij,bnj->bni", transform[:, :3, :3], markers)
    reconstructed += transform[:, None, :3, 3]
    np.testing.assert_allclose(reconstructed, observed, atol=1e-7)
    np.testing.assert_allclose(observed[0], observed[2], atol=1e-7)
    expected_determinant = -1 if hand == "left" else 1
    np.testing.assert_allclose(np.linalg.det(transform[:, :3, :3]), expected_determinant)
    np.testing.assert_array_equal(result["valid"], True)
    np.testing.assert_array_equal(result["mano_pose"], np.zeros((3, 48)))
    np.testing.assert_array_equal(result["mano_beta_mean"], np.arange(10))


def test_left_conversion_flips_only_normalized_marker_z():
    right, left = MarkerModel(), MarkerModel()
    convert_markers(right, marker_frames(1), handedness="right")
    convert_markers(left, marker_frames(1), handedness="left")
    torch.testing.assert_close(left.observed[0], right.observed[0] * torch.tensor([1, 1, -1]))


def test_incomplete_nonfinite_and_degenerate_frames_remain_invalid():
    markers = marker_frames(4)
    mask = np.ones((4, 21), bool)
    mask[1, 7] = False
    markers[2, 3] = np.nan
    markers[3] = 0
    model = MarkerModel()
    result = convert_markers(model, markers, handedness="right", valid_mask=mask)
    np.testing.assert_array_equal(result["valid"], [True, False, False, False])
    assert len(model.observed[0]) == 1
    assert np.isnan(result["mano_pose"][1:]).all()
    assert np.isnan(result["mano_beta"][1:]).all()
    assert np.isnan(result["world_to_canonical"][1:]).all()


def test_empty_and_all_invalid_inputs_do_not_run_network():
    for markers in (marker_frames(0), np.zeros((2, 21, 3), np.float32)):
        model = MarkerModel()
        result = convert_markers(model, markers, handedness="left")
        assert not model.observed
        assert not result["valid"].any()
        assert result["mano_pose"].shape == (len(markers), 48)
        assert np.isnan(result["mano_beta_mean"]).all()


def test_model_architecture_keeps_21_points_and_predicts_16_rotations():
    from egoemg.markers2mano.model import EfficientGraphTransformer4x

    model = EfficientGraphTransformer4x(embed_dim=32, num_layers=1, num_heads=4,
                                      lap_eig_k=4, dropout=0).eval()
    assert model.pose_head[0].in_features == 21 * 32
    assert not hasattr(model, "keep_index")
    with torch.inference_mode():
        pose, beta = model(torch.from_numpy(marker_frames(2)))
    assert pose.shape == (2, 16 * 6)
    assert beta.shape == (2, 10)


def test_checkpoint_loader_rejects_other_prediction_heads(tmp_path):
    checkpoint = tmp_path / "other.ckpt"
    torch.save({"hyper_parameters": MODEL_CONFIG, "state_dict": {
        "model.joint_head.0.weight": torch.zeros(1)}}, checkpoint)
    with pytest.raises(ValueError, match="global pose_head"):
        load_model(checkpoint)


def test_checkpoint_loader_rejects_wrong_model_width(tmp_path):
    checkpoint = tmp_path / "wrong.ckpt"
    torch.save({"hyper_parameters": {"hidden_dim": 32}, "state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="hidden_dim=1280"):
        load_model(checkpoint)


def test_cli_preserves_timestamps_and_refuses_hand_mismatch(monkeypatch, tmp_path):
    from egoemg.markers2mano import __main__ as cli

    monkeypatch.setattr(cli, "load_model", lambda *args: MarkerModel())
    source, destination = tmp_path / "markers.npz", tmp_path / "mano.npz"
    np.savez(source, markers_m=marker_frames(2), handedness="left",
             timestamps_s=[1.0, 1.01], source_rows=[3, 5])
    cli.main(["--input", str(source), "--output", str(destination), "--device", "cpu"])
    with np.load(destination) as result:
        np.testing.assert_array_equal(result["timestamps_s"], [1.0, 1.01])
        np.testing.assert_array_equal(result["source_rows"], [3, 5])
        assert result["handedness"].item() == "left"
        assert result["model_name"].item() == "markers2mano"
    with pytest.raises(SystemExit):
        cli.main(["--input", str(source), "--output", str(tmp_path / "other.npz"),
                  "--hand", "right"])
