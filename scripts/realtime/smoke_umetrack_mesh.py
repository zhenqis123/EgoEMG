#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_DATA_COLLECT_ROOT = _PROJECT_ROOT / "data_collect"
if _DATA_COLLECT_ROOT.exists() and str(_DATA_COLLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_COLLECT_ROOT))

from egoemg.realtime_local.mesh_visualizer import angles_to_umetrack_mesh


def main() -> None:
    import open3d as o3d

    mesh = angles_to_umetrack_mesh(np.zeros(22, dtype=np.float32))
    print(f"open3d {o3d.__version__}")
    print(f"vertices {mesh.vertices.shape} finite {np.isfinite(mesh.vertices).all()}")
    print(f"triangles {mesh.triangles.shape}")


if __name__ == "__main__":
    main()
