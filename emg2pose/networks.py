# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import collections
from collections.abc import Sequence

from typing import Literal

import torch

from torch import nn

##################
# TDS FEATURIZER #
##################


class Permute(nn.Module):
    """Permute the dimensions of the input tensor.
    For example:
    ```
    Permute('NTC', 'NCT') == x.permute(0, 2, 1)
    ```
    """

    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        assert len(from_dims) == len(
            to_dims
        ), "Same number of from- and to- dimensions should be specified for"

        if len(from_dims) not in {3, 4, 5, 6}:
            raise ValueError(
                "Only 3, 4, 5, and 6D tensors supported in Permute for now"
            )

        self.from_dims = from_dims
        self.to_dims = to_dims
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]

    def get_inverse_permute(self) -> "Permute":
        "Get the permute operation to get us back to the original dim order"
        return Permute(from_dims=self.to_dims, to_dims=self.from_dims)

    def __repr__(self):
        return f"Permute({self.from_dims!r} => {self.to_dims!r})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)


class BatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d except in NTC format"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.permute_forward = Permute("NTC", "NCT")
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        self.permute_back = Permute("NCT", "NTC")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.permute_back(self.bn(self.permute_forward(inputs)))


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
    ):
        """A 1D convolution with padding so the input and output lengths match."""

        super().__init__()

        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

        layers = {}
        layers["conv1d"] = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        if norm_type == "batch":
            layers["norm"] = BatchNorm1d(out_channels)

        layers["relu"] = nn.ReLU(inplace=True)
        layers["dropout"] = nn.Dropout(dropout)

        self.conv = nn.Sequential(
            *[layers[key] for key in layers if layers[key] is not None]
        )

        if norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type == "layer":
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()

        assert kernel_width % 2, "kernel_width must be odd."
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            dilation=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)

        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        B, C, T = inputs.shape  # BCT

        # BCT -> BcwT
        x = inputs.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(B, C, -1)  # BcwT -> BCT

        # Skip connection after downsampling
        T_out = x.shape[-1]
        x = x + inputs[..., -T_out:]

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = inputs
        x = x.swapaxes(-1, -2)  # BCT -> BTC
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)  # BTC -> BCT
        x += inputs

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)

        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            assert (
                num_features % channels == 0
            ), f"block_channels {channels} must evenly divide num_features {num_features}"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, feature_width, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class TdsStage(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        in_conv_kernel_width: int = 5,
        in_conv_stride: int = 1,
        num_blocks: int = 1,
        channels: int = 8,
        feature_width: int = 2,
        kernel_width: int = 1,
        out_channels: int | None = None,
    ):
        super().__init__()
        """Stage of several TdsBlocks preceded by a non-separable sub-sampling conv.

        The initial (and optionally sub-sampling) conv layer maps the number of
        input channels to the corresponding internal width used by the residual TDS
        blocks.

        Follows the multi-stage network construction from
        https://arxiv.org/abs/1904.02619.
        """

        layers: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()

        C = channels * feature_width

        self.out_channels = out_channels

        # Conv1d block
        if in_conv_kernel_width > 0:
            layers["conv1dblock"] = Conv1dBlock(
                in_channels,
                C,
                kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
            )
        elif in_channels != C:
            # Check that in_channels is consistent with TDS
            # channels and feature width
            raise ValueError(
                f"in_channels ({in_channels}) must equal channels *"
                f" feature_width ({channels} * {feature_width}) if"
                " in_conv_kernel_width is not positive."
            )

        # TDS block
        layers["tds_block"] = TDSConvEncoder(
            num_features=C,
            block_channels=[channels] * num_blocks,
            kernel_width=kernel_width,
        )

        # Linear projection
        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = self.layers(x)
        if self.out_channels is not None:
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TdsNetwork(nn.Module):
    def __init__(
        self, conv_blocks: Sequence[Conv1dBlock], tds_stages: Sequence[TdsStage]
    ):
        super().__init__()
        self.layers = nn.Sequential(*conv_blocks, *tds_stages)
        self.left_context = self._get_left_context(conv_blocks, tds_stages)
        self.right_context = 0

    def forward(self, x):
        return self.layers(x)

    def _get_left_context(self, conv_blocks, tds_stages) -> int:
        left, stride = 0, 1

        for conv_block in conv_blocks:
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

        for tds_stage in tds_stages:

            conv_block = tds_stage.layers.conv1dblock
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

            tds_block = tds_stage.layers.tds_block
            for _ in range(tds_block.num_blocks):
                left += (tds_block.kernel_width - 1) * stride

        return left


#############
# NEUROPOSE #
#############


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        max_pool_size: tuple[int, int],
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)
        maxpool = nn.MaxPool2d(kernel_size=max_pool_size, stride=max_pool_size)
        self.network = nn.Sequential(conv, bn, relu, dropout, maxpool)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        num_convs: int,
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        def _conv(in_channels: int, out_channels: int):
            """Single convolution block."""
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]

        modules = [*_conv(in_channels, out_channels)]
        for _ in range(num_convs - 1):
            modules += _conv(out_channels, out_channels)

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.network(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        upsampling: tuple[int, int],
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout_rate)
        scale_factor = (float(upsampling[0]), float(upsampling[1]))
        upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.network = nn.Sequential(conv, bn, relu, dropout, upsample)
        self.out_channels = out_channels

    def forward(self, x):
        return self.network(x)


class NeuroPose(nn.Module):
    def __init__(
        self,
        encoder_blocks: list[EncoderBlock],
        residual_blocks: list[ResidualBlock],
        decoder_blocks: list[DecoderBlock],
        linear_in_channels: int,
        out_channels: int = 22,
    ):
        super().__init__()
        self.network = nn.Sequential(*encoder_blocks, *residual_blocks, *decoder_blocks)
        self.linear = nn.Linear(linear_in_channels, out_channels)
        self.left_context = 0
        self.right_context = 0

    def forward(self, x):
        # Neuropose uses 2D convolutions over time and space, so we add a new
        # channel dimension corresponding to the network features.
        x = x[:, None].swapaxes(-1, -2)  # BCT -> BCtc
        x = self.network(x)
        x = x.swapaxes(-2, -3).flatten(-2)  # BCtc -> BTC
        return self.linear(x).swapaxes(-1, -2)  # BTC -> BCT


############
# DECODERS #
############


class MLP(nn.Module):
    """Basic MLP with optional scaling of the final output."""

    def __init__(
        self,
        in_channels: int,
        layer_sizes: list[int],
        out_channels: int,
        layer_norm: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()

        sizes = [in_channels] + layer_sizes
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if layer_norm:
                layers.append(nn.LayerNorm(out_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(sizes[-1], out_channels))

        self.mlp = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x):
        # x is (batch, channel)
        return self.mlp(x) * self.scale


class SequentialLSTM(nn.Module):
    """
    LSTM where each forward() call computes only a single time step, to be compatible
    looping over time manually.

    NOTE: Need to manually reset the state in outer context after each trajectory!
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_layers: int = 1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True)
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.mlp_out = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(hidden_size, out_channels)
        )
        self.scale = scale

    def reset_state(self):
        self.hidden = None

    def forward(self, x):
        """Forward pass for a single time step, where x is (batch, channel.)"""

        if self.hidden is None:
            # Initialize hidden state with zeros
            batch_size = x.size(0)
            device = x.device
            size = (self.num_layers, batch_size, self.hidden_size)
            self.hidden = (torch.zeros(*size).to(device), torch.zeros(*size).to(device))

        out, self.hidden = self.lstm(x[:, None], self.hidden)
        return self.mlp_out(out[:, 0]) * self.scale

    def _non_sequential_forward(self, x):
        """Non-sequential forward pass, where x is (batch, time, channel)."""
        return self.mlp_out(self.lstm(x)[0]) * self.scale


class CausalTransformerAutoregressiveDecoder(nn.Module):
    """Decoder-only causal Transformer (GPT-style) for autoregressive rollout.

    This module is designed to be a drop-in replacement for `SequentialLSTM` in
    `VEMG2PoseWithInitialState`-like loops:

    - Input: a single-step token `(B, token_dim)` (e.g. [emg_feature_t, prev_joint_pred])
    - Output: `(B, out_channels)` for that step
    - Keeps internal KV-cache via `reset_state()` + `use_cache=True`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        n_positions: int = 1024,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        output_scale: float = 1.0,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        GPT2Config, GPT2Model = _require_transformers()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.out_channels = out_channels
        self.output_scale = output_scale
        self.use_cache = use_cache

        self.in_proj = nn.Linear(in_channels, n_embd)
        self.model = GPT2Model(
            GPT2Config(
                # We always pass `inputs_embeds`, so the token embedding table (wte)
                # is never used. Keep it tiny to avoid wasting memory.
                vocab_size=1,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_positions=n_positions,
                n_ctx=n_positions,
                embd_pdrop=embd_pdrop,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
                layer_norm_epsilon=layer_norm_epsilon,
                use_cache=use_cache,
            )
        )
        # With `inputs_embeds`, GPT2Model will not touch `wte`, so freeze it to avoid
        # DDP "unused parameter" errors.
        if hasattr(self.model, "wte"):
            self.model.wte.requires_grad_(False)
        self.out_proj = nn.Linear(n_embd, out_channels)

        self._past_key_values = None

    def reset_state(self) -> None:
        self._past_key_values = None

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        if token.ndim == 2:
            inputs_embeds = self.in_proj(token).unsqueeze(1)  # (B, 1, H)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=self._past_key_values,
                use_cache=self.use_cache,
            )
            if self.use_cache:
                self._past_key_values = outputs.past_key_values

            hidden = outputs.last_hidden_state[:, 0, :]  # (B, H)
            return self.out_proj(hidden) * self.output_scale

        if token.ndim == 3:
            # Teacher-forcing / full-sequence forward. We disable caching because it
            # increases memory usage and provides no benefit in parallel decoding.
            inputs_embeds = self.in_proj(token)  # (B, T, H)
            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
            hidden = outputs.last_hidden_state  # (B, T, H)
            return self.out_proj(hidden) * self.output_scale  # (B, T, out)

        raise ValueError(
            "Expected token with shape (B, token_dim) or (B, T, token_dim), "
            f"got {tuple(token.shape)}"
        )


class RotaryCausalTransformerAutoregressiveDecoder(nn.Module):
    """Decoder-only causal Transformer using RoPE (rotary position embeddings).

    Implemented via `transformers.GPTNeoXModel` which applies rotary embeddings in
    attention, avoiding learned absolute position embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 256,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 2048,
        rotary_pct: float = 1.0,
        rotary_emb_base: int = 10_000,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        output_scale: float = 1.0,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        GPTNeoXConfig, GPTNeoXModel = _require_transformers_rope()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        self.out_channels = out_channels
        self.output_scale = output_scale
        self.use_cache = use_cache

        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.model = GPTNeoXModel(
            GPTNeoXConfig(
                # We always pass `inputs_embeds`, so token embeddings are unused.
                vocab_size=1,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                max_position_embeddings=max_position_embeddings,
                rotary_pct=rotary_pct,
                rotary_emb_base=rotary_emb_base,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
                use_cache=use_cache,
            )
        )
        # With `inputs_embeds`, GPTNeoXModel will not touch token embeddings.
        if hasattr(self.model, "embed_in"):
            self.model.embed_in.requires_grad_(False)

        self.out_proj = nn.Linear(hidden_size, out_channels)
        self._past_key_values = None

    def reset_state(self) -> None:
        self._past_key_values = None

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        if token.ndim == 2:
            inputs_embeds = self.in_proj(token).unsqueeze(1)  # (B, 1, H)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=self._past_key_values,
                use_cache=self.use_cache,
            )
            if self.use_cache:
                self._past_key_values = outputs.past_key_values
            hidden = outputs.last_hidden_state[:, 0, :]  # (B, H)
            return self.out_proj(hidden) * self.output_scale

        if token.ndim == 3:
            inputs_embeds = self.in_proj(token)  # (B, T, H)
            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
            hidden = outputs.last_hidden_state  # (B, T, H)
            return self.out_proj(hidden) * self.output_scale  # (B, T, out)

        raise ValueError(
            "Expected token with shape (B, token_dim) or (B, T, token_dim), "
            f"got {tuple(token.shape)}"
        )


class WindowTransformerCrossAttnLastStepDecoder(nn.Module):
    """Legacy last-step cross-attention decoder (kept for backward compatibility)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 20,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        query_token_dim: int = 20,
        num_query_tokens: int = 20,
        variant: str | None = None,
        presets: dict | None = None,
    ) -> None:
        super().__init__()
        BertConfig, BertModel = _require_transformers_bert()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads, got "
                f"{hidden_size} and {num_attention_heads}"
            )
        if query_token_dim <= 0:
            raise ValueError(f"query_token_dim must be positive, got {query_token_dim}")
        if num_query_tokens <= 0:
            raise ValueError(f"num_query_tokens must be positive, got {num_query_tokens}")
        if out_channels != num_query_tokens:
            raise ValueError(
                f"out_channels ({out_channels}) must equal num_query_tokens ({num_query_tokens}) "
                "to map one query to one angle."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_query_tokens = num_query_tokens

        self.in_proj = nn.Linear(in_channels, hidden_size)

        enc_cfg = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            is_decoder=False,
            add_cross_attention=False,
        )
        self.encoder = BertModel(enc_cfg)

        dec_cfg = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=decoder_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            is_decoder=True,
            add_cross_attention=True,
        )
        self.decoder = BertModel(dec_cfg)

        # We always pass `inputs_embeds`, so word embeddings are unused; freeze them
        # to avoid DDP "unused parameter" errors.
        self.encoder.embeddings.word_embeddings.requires_grad_(False)
        self.decoder.embeddings.word_embeddings.requires_grad_(False)

        # Learnable queries: initialize with small normal noise (BERT-style ~N(0, 0.02))
        self.query_tokens = nn.Parameter(torch.empty(num_query_tokens, query_token_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.query_proj = nn.Linear(query_token_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape (B, C, T), got {tuple(features.shape)}"
            )
        batch_size, channels, time = features.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected features C={self.in_channels}, got C={channels}"
            )
        if time <= 0:
            raise ValueError(f"Expected T>0, got T={time}")

        x = features.swapaxes(-1, -2)
        x = self.in_proj(x)  # (B, T, H)
        attn_mask = torch.ones(batch_size, time, device=features.device, dtype=torch.long)
        memory = self.encoder(inputs_embeds=x, attention_mask=attn_mask).last_hidden_state

        query = self.query_proj(self.query_tokens).unsqueeze(0).expand(
            batch_size, self.num_query_tokens, -1
        )
        query_mask = torch.ones(
            batch_size, self.num_query_tokens, device=features.device, dtype=torch.long
        )
        dec_out = self.decoder(
            inputs_embeds=query,
            attention_mask=query_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=attn_mask,
        ).last_hidden_state  # (B, num_queries, H)

        return self.out_proj(dec_out).squeeze(-1)  # (B, num_queries)


class Emg2PoseFormerDecoder(nn.Module):
    """Unified Transformer decoder with selectable supervision modes.

    - prediction_mode="last": encoder + cross-attn readout tokens -> (B, out_channels)
    - prediction_mode="full": per-timestep outputs -> (B, out_channels, T); optional causal self-attn.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 20,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        query_token_dim: int = 128,
        num_query_tokens: int = 20,
        prediction_mode: str = "last",
        causal_self_attention: bool = False,
    ) -> None:
        super().__init__()
        BertConfig, BertModel = _require_transformers_bert()

        if prediction_mode not in ("last", "full"):
            raise ValueError(f"prediction_mode must be 'last' or 'full', got {prediction_mode}")
        self.prediction_mode = prediction_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_query_tokens = num_query_tokens
        self.causal_self_attention = causal_self_attention

        self.in_proj = nn.Linear(in_channels, hidden_size)

        enc_cfg = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            is_decoder=False,
            add_cross_attention=False,
        )

        dec_cfg = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=decoder_layers,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            is_decoder=True,
            add_cross_attention=True,
        )

        full_dec_cfg = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=encoder_layers,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            is_decoder=True,
            add_cross_attention=False,
        )

        self.encoder = BertModel(enc_cfg)
        self.decoder = BertModel(dec_cfg)
        self.full_decoder = BertModel(full_dec_cfg)

        for mdl in (self.encoder, self.decoder, self.full_decoder):
            mdl.embeddings.word_embeddings.requires_grad_(False)

        self.query_tokens = nn.Parameter(torch.empty(num_query_tokens, query_token_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.query_proj = nn.Linear(query_token_dim, hidden_size)
        self.out_proj_last = nn.Linear(hidden_size, 1)
        self.out_proj_full = nn.Linear(hidden_size, out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected features with shape (B, C, T), got {tuple(features.shape)}")
        batch_size, channels, time = features.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected features C={self.in_channels}, got C={channels}")

        x = self.in_proj(features.swapaxes(-1, -2))  # (B, T, H)
        attn_mask = torch.ones(batch_size, time, device=features.device, dtype=torch.long)

        if self.prediction_mode == "last":
            memory = self.encoder(inputs_embeds=x, attention_mask=attn_mask).last_hidden_state
            query = self.query_proj(self.query_tokens).unsqueeze(0).expand(
                batch_size, self.num_query_tokens, -1
            )
            query_mask = torch.ones(
                batch_size, self.num_query_tokens, device=features.device, dtype=torch.long
            )
            dec_out = self.decoder(
                inputs_embeds=query,
                attention_mask=query_mask,
                encoder_hidden_states=memory,
                encoder_attention_mask=attn_mask,
            ).last_hidden_state
            return self.out_proj_last(dec_out).squeeze(-1)  # (B, num_queries)

        # prediction_mode == "full"
        if self.causal_self_attention:
            seq_out = self.full_decoder(inputs_embeds=x, attention_mask=attn_mask).last_hidden_state
        else:
            seq_out = self.encoder(inputs_embeds=x, attention_mask=attn_mask).last_hidden_state

        out = self.out_proj_full(seq_out)  # (B, T, out_channels)
        return out.swapaxes(-1, -2)  # (B, out_channels, T)
