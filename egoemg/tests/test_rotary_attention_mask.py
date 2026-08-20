"""Regression tests for the rotary-attention SDPA mask semantics fix.

nn.MultiheadAttention treats a bool mask as True=masked, while
F.scaled_dot_product_attention treats it as True=allowed. The mask must be
inverted on the way into SDPA, otherwise causal decoding attends only to the
FUTURE and the last position is fully masked (NaN after softmax).
"""
from __future__ import annotations

import torch

from egoemg.models.decoders.transformer import RotaryMultiheadAttention


def _attn() -> RotaryMultiheadAttention:
    torch.manual_seed(0)
    attn = RotaryMultiheadAttention(embed_dim=16, num_heads=4)
    attn.eval()
    return attn


def test_causal_mask_attends_only_to_past():
    attn = _attn()
    T = 6
    x = torch.randn(2, T, 16)
    mask = torch.ones(T, T, dtype=torch.bool).triu(1)  # MHA semantics: True=forbidden
    out = attn(x, x, x, attn_mask=mask, need_weights=False)[0]
    assert torch.isfinite(out).all(), "causal mask must not fully mask any position"

    # Perturbing token t must not change outputs at positions < t.
    x2 = x.clone()
    x2[:, 3] += 10.0
    out2 = attn(x2, x2, x2, attn_mask=mask, need_weights=False)[0]
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-5), "future leakage into past positions"
    assert not torch.allclose(out[:, 3], out2[:, 3], atol=1e-5), "position 3 must see its own change"


def test_key_padding_mask_ignores_padded_positions():
    attn = _attn()
    B, T = 2, 5
    x = torch.randn(B, T, 16)
    # MHA semantics: True = padding (forbidden). Pad the last two steps.
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[:, -2:] = True
    out = attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]
    assert torch.isfinite(out).all(), "unpadded rows must never be fully masked"

    x2 = x.clone()
    x2[:, -2:] += 10.0  # change only padded content
    out2 = attn(x2, x2, x2, key_padding_mask=pad, need_weights=False)[0]
    assert torch.allclose(
        out[:, :-2], out2[:, :-2], atol=1e-5
    ), "padded keys must not influence outputs at unpadded positions"


def test_causal_transformer_decoder_is_finite_and_causal():
    from egoemg.models.decoders.transformer import TransformerDecoder

    torch.manual_seed(0)
    dec = TransformerDecoder(
        in_channels=8,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        pos_encoding="rope",
        causal=True,
    )
    dec.eval()
    x = torch.randn(2, 8, 7)
    out = dec(x)
    assert torch.isfinite(out).all()

    x2 = x.clone()
    x2[:, :, 4] += 5.0
    out2 = dec(x2)
    assert torch.allclose(out[:, :, :4], out2[:, :, :4], atol=1e-4), "future leakage in decoder"
