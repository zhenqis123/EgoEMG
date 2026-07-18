import numpy as np
import torch

from emg2pose.realtime_local.buffer import SlidingWindowBuffer
from emg2pose.realtime_local.pipeline import LocalSmallStreamer
from emg2pose.realtime_local.serial import SerialProtocol
from emg2pose.realtime_local.small_model import (
    SMALL_CHANNEL_POSITIONS,
    map_small_channels,
)


def _encode_int24(value: int) -> bytes:
    if value < 0:
        value += 1 << 24
    return bytes([(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])


def test_small_channel_map_zero_fills_16ch_layout():
    x = np.arange(24, dtype=np.float32).reshape(3, 8)

    y = map_small_channels(x)

    assert y.shape == (3, 16)
    np.testing.assert_array_equal(y[:, SMALL_CHANNEL_POSITIONS], x)
    empty_positions = sorted(set(range(16)) - set(SMALL_CHANNEL_POSITIONS.tolist()))
    np.testing.assert_array_equal(y[:, empty_positions], 0.0)


def test_sliding_window_buffer_emits_latest_ordered_window():
    buf = SlidingWindowBuffer(window_length=5, stride=2, n_channels=1)
    buf.push(np.arange(4, dtype=np.float32).reshape(-1, 1))
    assert not buf.has_window()

    buf.push(np.asarray([[4], [5]], dtype=np.float32))

    assert buf.has_window()
    np.testing.assert_array_equal(
        buf.get_window(),
        np.asarray([[1], [2], [3], [4], [5]], dtype=np.float32),
    )
    assert not buf.has_window()


def test_sliding_window_buffer_can_drop_stale_windows_but_emit_latest():
    buf = SlidingWindowBuffer(window_length=5, stride=2, n_channels=1)
    buf.push(np.arange(10, dtype=np.float32).reshape(-1, 1))

    assert buf.ready_count() > 1
    buf.keep_latest_ready()

    assert buf.ready_count() == 1
    np.testing.assert_array_equal(
        buf.get_window(),
        np.asarray([[5], [6], [7], [8], [9]], dtype=np.float32),
    )


def test_serial_protocol_decodes_8ch_int24_emg_packet():
    proto = SerialProtocol(
        header=b"\xAA\x55\xAA",
        packet_len=29,
        emg_type=0xAA,
        imu_type=0xBB,
    )
    values = [0, 1, -1, 127, -128, 32767, -32768, 123456]
    payload = b"".join(_encode_int24(v) for v in values)
    packet = proto.header + bytes([proto.emg_type, 0]) + payload

    decoded = proto.decode_emg_packet(packet)

    assert decoded is not None
    np.testing.assert_array_equal(decoded, np.asarray(values, dtype=np.float32))


def test_predict_window_uses_default_half_second_delay(monkeypatch, tmp_path):
    class DummyModel:
        left_context = 510

        def __call__(self, batch):
            t = 230
            return torch.arange(t, dtype=torch.float32).view(1, 1, t).expand(1, 22, t)

    def fake_load(*_args, **_kwargs):
        return DummyModel()

    monkeypatch.setattr("emg2pose.realtime_local.pipeline.load_small_emgformer", fake_load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    streamer = LocalSmallStreamer(
        checkpoint_path=tmp_path / "dummy.pt",
        device="cpu",
        output_delay_s=0.5,
    )
    pred = streamer.predict_window(
        np.zeros((12000, 8), dtype=np.float32),
        timestamp=100.0,
        sample_index=12000,
    )

    low_rate = torch.arange(230, dtype=torch.float32).view(1, 1, 230)
    expected = torch.nn.functional.interpolate(
        low_rate,
        size=11490,
        mode="linear",
    )[0, 0, 10489].item()
    assert abs(float(pred.angles[0]) - expected) < 1e-5
    assert pred.timestamp == 99.5


def test_umetrack_mesh_fk_shape():
    from emg2pose.realtime_local.mesh_visualizer import angles_to_umetrack_mesh

    mesh = angles_to_umetrack_mesh(np.zeros(22, dtype=np.float32))

    assert mesh.vertices.ndim == 2
    assert mesh.vertices.shape[1] == 3
    assert mesh.triangles.ndim == 2
    assert mesh.triangles.shape[1] == 3
    assert np.isfinite(mesh.vertices).all()


def test_umetrack_mesh_fk_matches_batch_ik_mesh_path():
    from emg2pose.kinematics import (
        apply_to_hand_model,
        broadcast_hand_model_to,
        load_default_hand_model,
    )
    from emg2pose.realtime_local.mesh_visualizer import UmeTrackMeshForwarder
    from emg2pose.UmeTrack.lib.common.hand_skinning import (
        _get_skinned_vertices,
        _hand_skinning_transform,
        _lbs,
    )

    angles = np.linspace(-0.5, 0.5, 22, dtype=np.float32)
    hand_model = load_default_hand_model()
    hm = broadcast_hand_model_to(hand_model, (1,))
    hm = apply_to_hand_model(hm, lambda t: t.float())
    wrist_tf = torch.eye(4).unsqueeze(0)
    a = torch.from_numpy(angles).reshape(1, -1)
    skin_xfs = _hand_skinning_transform(
        hm.joint_rotation_axes.reshape(1, -1, 3),
        hm.joint_rest_positions.reshape(1, -1, 3),
        a,
        wrist_tf,
    )
    weights = hm.dense_bone_weights.reshape(1, -1, 17)
    rest_vertices = hm.mesh_vertices.reshape(1, -1, 3)
    skinned_vertices = _get_skinned_vertices(rest_vertices, weights)
    expected = _lbs(skin_xfs, skinned_vertices)[..., :3][0].numpy()

    actual = UmeTrackMeshForwarder()(angles).vertices

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
