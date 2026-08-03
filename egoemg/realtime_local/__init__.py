"""Minimal local streaming runtime for EMGFormer deployment."""

from egoemg.realtime_local.buffer import SlidingWindowBuffer
from egoemg.realtime_local.mano_mapper import RuntimeManoToUmeTrackMapper
from egoemg.realtime_local.online_full_adapt import FullModelOnlineTrainer
from egoemg.realtime_local.pipeline import LocalSmallStreamer, Prediction
from egoemg.realtime_local.preprocess import SmallPreprocessor
from egoemg.realtime_local.small_model import (
    SMALL_CHANNEL_POSITIONS,
    load_small_emgformer,
    map_small_channels,
)

__all__ = [
    "LocalSmallStreamer",
    "Prediction",
    "FullModelOnlineTrainer",
    "RuntimeManoToUmeTrackMapper",
    "SMALL_CHANNEL_POSITIONS",
    "SlidingWindowBuffer",
    "SmallPreprocessor",
    "load_small_emgformer",
    "map_small_channels",
]
