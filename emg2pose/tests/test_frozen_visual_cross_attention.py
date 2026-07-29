import torch

from emg2pose.models.modules.mid_fusion import (
    FrozenVisualCrossAttention,
    make_invalid_anchor_emg,
)


def test_frozen_visual_cross_attention_shape_and_backward() -> None:
    module = FrozenVisualCrossAttention(
        emg_dim=256,
        layer3_dim=1024,
        layer4_dim=2048,
        fusion_dim=256,
        num_heads=8,
        ffn_dim=512,
        dropout=0.0,
    )
    emg = torch.randn(2, 256, 23, requires_grad=True)
    layer3 = torch.randn(2, 1024, 8, 8)
    layer4 = torch.randn(2, 2048, 4, 4)

    output = module(emg, layer3, layer4, torch.tensor([True, False]))

    assert output.shape == emg.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert emg.grad is not None
    assert module.layer3_proj.weight.grad is not None
    assert module.cross_attention.out_proj.weight.grad is not None


def test_two_layer_visual_cross_attention_shape_and_backward() -> None:
    module = FrozenVisualCrossAttention(
        emg_dim=32,
        layer3_dim=48,
        layer4_dim=64,
        fusion_dim=32,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        num_layers=2,
    )
    emg = torch.randn(2, 32, 11, requires_grad=True)
    output = module(
        emg,
        torch.randn(2, 48, 6, 6),
        torch.randn(2, 64, 3, 3),
        torch.tensor([True, True]),
    )
    assert output.shape == emg.shape
    output.square().mean().backward()
    assert module.extra_blocks[0].cross_attention.out_proj.weight.grad is not None


def test_invalid_anchor_emg_uses_same_hand_mismatches() -> None:
    emg = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
    hands = torch.tensor([0, 0, 1, 1])

    anchor = make_invalid_anchor_emg(emg, hands, shuffle_fraction=1.0)

    assert torch.equal(anchor[:, 0, 0], torch.tensor([1.0, 0.0, 3.0, 2.0]))


def test_invalid_anchor_emg_zero_mode_and_fraction_validation() -> None:
    emg = torch.randn(3, 2, 5)
    assert torch.count_nonzero(
        make_invalid_anchor_emg(emg, None, shuffle_fraction=0.0)
    ) == 0
    try:
        make_invalid_anchor_emg(emg, None, shuffle_fraction=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid shuffle fraction must raise ValueError")
