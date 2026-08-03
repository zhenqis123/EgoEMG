#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy.io as sio

import egoemg.visualization as visualization


NINAPRO_TO_EMG2POSE = [
    "CMC1_f",  # THUMB_CMC_FE
    "CMC1_a",  # THUMB_CMC_AA
    "MCP1",  # THUMB_MCP_FE
    "IP1",  # THUMB_IP_FE
    "MCP2_a",  # INDEX_MCP_AA (may be NaN)
    "MCP2_f",  # INDEX_MCP_FE
    "PIP2",  # INDEX_PIP_FE
    "DIP2",  # INDEX_DIP_FE
    None,  # MIDDLE_MCP_AA (not available in Ninapro angles)
    "MCP3_f",  # MIDDLE_MCP_FE
    "PIP3",  # MIDDLE_PIP_FE
    "DIP3",  # MIDDLE_DIP_FE
    "MCP4_a",  # RING_MCP_AA
    "MCP4_f",  # RING_MCP_FE
    "PIP4",  # RING_PIP_FE
    "DIP4",  # RING_DIP_FE
    "MCP5_a",  # PINKY_MCP_AA
    "MCP5_f",  # PINKY_MCP_FE
    "PIP5",  # PINKY_PIP_FE
    "DIP5",  # PINKY_DIP_FE
]

# Per-joint sign control for emg2pose ordering (20 dims).
# Use 1 for keep, -1 to flip. Adjust as needed.
EMG2POSE_SIGN = np.array(
    [
        1,  # THUMB_CMC_FE
        1,  # THUMB_CMC_AA
        1,  # THUMB_MCP_FE
        1,  # THUMB_IP_FE
        1,  # INDEX_MCP_AA
        1,  # INDEX_MCP_FE
        1,  # INDEX_PIP_FE
        -1,  # INDEX_DIP_FE (match test_glove in_j3 sign flip)
        1,  # MIDDLE_MCP_AA (filled as 0)
        1,  # MIDDLE_MCP_FE
        1,  # MIDDLE_PIP_FE
        -1,  # MIDDLE_DIP_FE
        1,  # RING_MCP_AA
        1,  # RING_MCP_FE
        1,  # RING_PIP_FE
        -1,  # RING_DIP_FE
        1,  # PINKY_MCP_AA
        1,  # PINKY_MCP_FE
        1,  # PINKY_PIP_FE
        -1,  # PINKY_DIP_FE (match test_glove pi_j3 sign flip)
    ],
    dtype=np.float32,
)

# Per-joint offset in degrees for emg2pose ordering (20 dims).
# Default applies test_glove-style offsets for non-thumb joints.
EMG2POSE_OFFSET_DEG = np.array(
    [
        0.0,  # THUMB_CMC_FE
        0.0,  # THUMB_CMC_AA
        0.0,  # THUMB_MCP_FE
        0.0,  # THUMB_IP_FE
        0.0,  # INDEX_MCP_AA
        0.0,  # INDEX_MCP_FE
        0.0,  # INDEX_PIP_FE
        0.0,  # INDEX_DIP_FE
        0.0,  # MIDDLE_MCP_AA (filled as 0)
        -5.73,  # MIDDLE_MCP_FE (mi_j1 - 0.1 rad)
        0.0,  # MIDDLE_PIP_FE
        -28.65,  # MIDDLE_DIP_FE (mi_j3 - 0.5 rad)
        -8.59,  # RING_MCP_AA (ri_j0 - 0.15 rad)
        0.0,  # RING_MCP_FE
        0.0,  # RING_PIP_FE
        -28.65,  # RING_DIP_FE (ri_j3 - 0.5 rad)
        -17.19,  # PINKY_MCP_AA (pi_j0 - 0.3 rad)
        -17.19,  # PINKY_MCP_FE (pi_j1 - 0.3 rad)
        0.0,  # PINKY_PIP_FE
        5.73,  # PINKY_DIP_FE (pi_j3 + 0.1 rad after sign flip)
    ],
    dtype=np.float32,
)
# EMG2POSE_OFFSET_DEG = np.array(
#     [
#         0.0,  # THUMB_CMC_FE
#         0.0,  # THUMB_CMC_AA
#         0.0,  # THUMB_MCP_FE
#         0.0,  # THUMB_IP_FE
#         0.0,  # INDEX_MCP_AA
#         0.0,  # INDEX_MCP_FE
#         0.0,  # INDEX_PIP_FE
#         0.0,  # INDEX_DIP_FE
#         0.0,  # MIDDLE_MCP_AA (filled as 0)
#         0,  # MIDDLE_MCP_FE (mi_j1 - 0.1 rad)
#         0.0,  # MIDDLE_PIP_FE
#         0,  # MIDDLE_DIP_FE (mi_j3 - 0.5 rad)
#         0,  # RING_MCP_AA (ri_j0 - 0.15 rad)
#         0.0,  # RING_MCP_FE
#         0.0,  # RING_PIP_FE
#         -28.65,  # RING_DIP_FE (ri_j3 - 0.5 rad)
#         -17.19,  # PINKY_MCP_AA (pi_j0 - 0.3 rad)
#         -17.19,  # PINKY_MCP_FE (pi_j1 - 0.3 rad)
#         0.0,  # PINKY_PIP_FE
#         5.73,  # PINKY_DIP_FE (pi_j3 + 0.1 rad after sign flip)
#     ],
#     dtype=np.float32,
# )


def _load_mat(path: Path) -> dict:
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _normalize_label(label: str) -> str:
    label = str(label).strip()
    if ":" in label:
        label = label.split(":", 1)[1].strip()
    return label


def _get_order_labels(mat: dict, num_channels: int) -> list[str]:
    order = mat.get("order_of_angles")
    if order is None:
        return [f"ch{i+1}" for i in range(num_channels)]
    order_list = np.atleast_1d(order).tolist()
    return [_normalize_label(x) for x in order_list]


def _get_series(mat: dict, key: str) -> np.ndarray | None:
    if key not in mat:
        return None
    arr = np.asarray(mat[key]).squeeze()
    if arr.ndim == 0:
        return None
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _extract_emg2pose_angles(angles: np.ndarray, order_labels: list[str]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(order_labels)}
    out = np.zeros((angles.shape[0], len(NINAPRO_TO_EMG2POSE)), dtype=np.float32)
    for out_idx, label in enumerate(NINAPRO_TO_EMG2POSE):
        if label is None:
            continue
        if label not in label_to_idx:
            raise KeyError(f"Missing Ninapro angle label: {label}")
        col = angles[:, label_to_idx[label]]
        if not np.isfinite(col).any():
            continue
        out[:, out_idx] = col
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Ninapro angles using egoemg/UmeTrack hand model."
    )
    parser.add_argument(
        "--mat",
        type=Path,
        required=True,
        help="Path to Ninapro .mat containing 'angles'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ninapro_hand.html"),
        help="Output HTML path.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--stop", type=int, default=-1, help="Stop frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Disable sign/offset correction; visualize raw angles.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plotly window (requires GUI).",
    )
    args = parser.parse_args()

    mat = _load_mat(args.mat)
    if "angles" not in mat:
        raise KeyError("Missing 'angles' in Ninapro .mat file.")
    print(mat.keys())
    print("stimulus", mat['stimulus'])
    print("restimulus", mat['restimulus'])
    print("exercise", mat['exercise'])
    angles = mat["angles"]
    if angles.ndim != 2:
        raise ValueError(f"Expected 2D angles array, got {angles.shape}.")

    stimulus = _get_series(mat, "stimulus")
    restimulus = _get_series(mat, "restimulus")
    total_len = angles.shape[0]
    if stimulus is not None:
        total_len = min(total_len, stimulus.shape[0])
    if restimulus is not None:
        total_len = min(total_len, restimulus.shape[0])
    if total_len != angles.shape[0]:
        angles = angles[:total_len]
        if stimulus is not None:
            stimulus = stimulus[:total_len]
        if restimulus is not None:
            restimulus = restimulus[:total_len]

    order_labels = _get_order_labels(mat, angles.shape[1])
    emg2pose_angles_deg = _extract_emg2pose_angles(angles, order_labels)

    if args.stop <= 0:
        args.stop = emg2pose_angles_deg.shape[0]
    sel = slice(args.start, args.stop, args.stride)
    emg2pose_angles_deg = emg2pose_angles_deg[sel]
    if stimulus is not None:
        stimulus = stimulus[sel]
    if restimulus is not None:
        restimulus = restimulus[sel]
    if not args.no_correction:
        if EMG2POSE_SIGN.shape[0] != emg2pose_angles_deg.shape[1]:
            raise ValueError(
                f"EMG2POSE_SIGN has {EMG2POSE_SIGN.shape[0]} entries, "
                f"expected {emg2pose_angles_deg.shape[1]}."
            )
        if EMG2POSE_OFFSET_DEG.shape[0] != emg2pose_angles_deg.shape[1]:
            raise ValueError(
                f"EMG2POSE_OFFSET_DEG has {EMG2POSE_OFFSET_DEG.shape[0]} entries, "
                f"expected {emg2pose_angles_deg.shape[1]}."
            )
        emg2pose_angles_deg = emg2pose_angles_deg * EMG2POSE_SIGN[None, :]
        emg2pose_angles_deg = emg2pose_angles_deg + EMG2POSE_OFFSET_DEG[None, :]
    emg2pose_angles_rad = np.deg2rad(emg2pose_angles_deg)

    fig = visualization.get_plotly_animation_for_joint_angles(emg2pose_angles_rad)
    if stimulus is not None or restimulus is not None:
        def _frame_text(i: int) -> str:
            parts = []
            if stimulus is not None:
                parts.append(f"stimulus={int(stimulus[i])}")
            if restimulus is not None:
                parts.append(f"restimulus={int(restimulus[i])}")
            return " | ".join(parts)

        if fig.frames:
            for i, frame in enumerate(fig.frames):
                frame.layout = go.Layout(
                    annotations=[
                        dict(
                            text=_frame_text(i),
                            x=0.01,
                            y=0.99,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=14),
                        )
                    ]
                )
            fig.update_layout(
                annotations=[
                    dict(
                        text=_frame_text(0),
                        x=0.01,
                        y=0.99,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14),
                    )
                ]
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output))
    if args.show:
        fig.show()


if __name__ == "__main__":
    pio.renderers.default = "browser"
    main()
