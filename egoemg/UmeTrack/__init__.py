"""Vendored UmeTrack hand-model/tracking code.

The vendored tree keeps its original repo-root-absolute imports
(``import lib...``, ``from lib.common...``). Register this copy's root on
``sys.path`` so those imports resolve to the vendored files without the
legacy separate editable install of the ``umetrack`` subdirectory.
"""
import sys
from pathlib import Path

_UMETRACK_ROOT = str(Path(__file__).resolve().parent)
if _UMETRACK_ROOT not in sys.path:
    sys.path.insert(0, _UMETRACK_ROOT)
