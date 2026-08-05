# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import io
import json
import pathlib
from joblib import delayed
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm
from PIL import Image

from emg2pose.UmeTrack.lib.tracker.video_pose_data import load_hand_model_from_dict
from emg2pose.UmeTrack.lib.common.hand import HandModel
from emg2pose.UmeTrack.lib.common.hand_skinning import _skin_points
from emg2pose.utils import ProgressParallel


DEFAULT_RANGES = {"x": [-20, 200], "y": [-100, 40], "z": [-70, 100]}


def load_hand_model_from_json(filepath: str) -> HandModel:
    with open(filepath, "rb") as fp:
        hand_model_dict = json.load(fp)
    hand_model = load_hand_model_from_dict(hand_model_dict)
    return hand_model


def load_default_hand_model():
    umetrack_dir = pathlib.Path(__file__).parent / "UmeTrack"
    default_hand_model_path = umetrack_dir / "dataset" / "generic_hand_model.json"
    return load_hand_model_from_json(default_hand_model_path)


def mirror_profile(profile: HandModel) -> HandModel:
    mirrored_joint_rotation_axes = profile.joint_rotation_axes.clone()
    mirrored_joint_rest_positions = profile.joint_rest_positions.clone()
    mirrored_landmark_rest_positions = profile.landmark_rest_positions.clone()
    mirrored_mesh_vertices = None

    mirrored_joint_rotation_axes[..., 1:] *= -1
    mirrored_joint_rest_positions[..., 0] *= -1
    mirrored_landmark_rest_positions[..., 0] *= -1
    if profile.mesh_vertices is not None:
        mirrored_mesh_vertices = profile.mesh_vertices.clone()
        mirrored_mesh_vertices[..., 0] *= -1
    return profile._replace(
        joint_rotation_axes=mirrored_joint_rotation_axes,
        joint_rest_positions=mirrored_joint_rest_positions,
        landmark_rest_positions=mirrored_landmark_rest_positions,
        mesh_vertices=mirrored_mesh_vertices,
    )


def skin_vertices(
    profile: HandModel,
    joint_angles: torch.Tensor,
    wrist_transforms: torch.Tensor | None = None,
) -> torch.Tensor:

    assert profile.mesh_vertices is not None, "mesh vertices should not be none"
    assert (
        profile.dense_bone_weights is not None
    ), "dense bone weights should not be none"

    if wrist_transforms is None:
        leading_dims = joint_angles.shape[1:]
        affine_identity = torch.eye(4, device=joint_angles.device)
        wrist_transforms = torch.broadcast_to(
            affine_identity, leading_dims + affine_identity.shape
        )

    vertices = _skin_points(
        profile.joint_rest_positions,
        profile.joint_rotation_axes,
        profile.dense_bone_weights,
        joint_angles,
        profile.mesh_vertices,
        wrist_transforms,
    )

    leading_dims = joint_angles.shape[:-1]
    vertices = vertices.reshape(list(leading_dims) + list(vertices.shape[-2:]))
    return vertices


def skin_vertices_np(
    profile: HandModel,
    joint_angles: np.ndarray,
    wrist_transforms: np.ndarray | None = None,
) -> np.ndarray:
    vertices = skin_vertices(
        profile,
        torch.from_numpy(joint_angles).float(),
        (
            None
            if wrist_transforms is None
            else torch.from_numpy(wrist_transforms).float()
        ),
    )
    return vertices.numpy()


def skin_mesh_from_angles(
    joint_angles,
    user_profile=None,
    flip=False,
):
    if user_profile is None:
        user_profile = load_default_hand_model()
    if flip:
        user_profile = mirror_profile(user_profile)

    vertices = skin_vertices_np(user_profile, joint_angles)
    triangles = user_profile.mesh_triangles
    return vertices, triangles


def get_default_lighting():
    lighting = dict(ambient=0.85, diffuse=0.2, specular=0.5, roughness=1.0)
    lightposition = dict(x=10, y=-500, z=-1)
    return lighting, lightposition


def generate_hand_mesh_from_joint_angles(
    joint_angles, color="lightpink", flip=False, opacity=0.5, **mesh_kwargs
):
    lighting, lightposition = get_default_lighting()

    vertices, triangles = skin_mesh_from_angles(joint_angles, flip=flip)
    x, y, z = vertices.T
    i, j, k = triangles.T

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        lighting=lighting,
        lightposition=lightposition,
        **mesh_kwargs,
    )


def _plot_hand_mesh_from_angles(
    joint_angles,
    color="lightpink",
    opacity=1.0,
    flip=False,
    show_triangles=False,
    **mesh_kwargs,
):

    lighting, lightposition = get_default_lighting()

    vertices, triangles = skin_mesh_from_angles(joint_angles, flip=flip)
    x, y, z = vertices.T
    i, j, k = triangles.T

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        lighting=lighting,
        lightposition=lightposition,
        **mesh_kwargs,
    )

    fig = go.Figure()
    fig.add_trace(mesh)

    if show_triangles:
        tri_points = vertices[triangles.int()]
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k % 3][0] for k in range(4)] + [None])
            Ye.extend([T[k % 3][1] for k in range(4)] + [None])
            Ze.extend([T[k % 3][2] for k in range(4)] + [None])

        # define the trace for triangle sides
        lines = go.Scatter3d(
            x=Xe,
            y=Ye,
            z=Ze,
            mode="lines",
            name="",
            line=dict(color="rgb(70,70,70)", width=2.0),
        )
        fig.add_trace(lines)

    return fig


def _set_3d_plot_layout(
    fig: go.Figure, auto_range=False, flip=False, clean_background=True
):

    xmin, xmax = DEFAULT_RANGES["x"]
    ymin, ymax = DEFAULT_RANGES["y"]
    zmin, zmax = DEFAULT_RANGES["z"]

    if flip:
        xmin *= -1
        xmax *= -1
        xmin, xmax = xmax, xmin

    if auto_range:
        mesh = None
        for data in fig.data:
            if isinstance(data, go.Mesh3d):
                mesh = fig.data[0]
                break

        if mesh is None:
            raise ValueError("Can't find mesh in provided fig.")

        xmax, xmin = np.max(mesh.x), np.min(mesh.x)
        ymax, ymin = np.max(mesh.y), np.min(mesh.y)
        zmax, zmin = np.max(mesh.z), np.min(mesh.z)

    camera = {
        "eye": {"x": -0.25, "y": -0.9, "z": 0.3},
        "projection": {"type": "perspective"},
    }

    addl_kwargs = dict()
    if clean_background:
        addl_kwargs = dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
            showgrid=False,
            title="",
        )

    xrange = xmax - xmin
    yrange = ymax - ymin
    zrange = zmax - zmin

    xratio = xrange / xrange
    yratio = yrange / xrange
    zratio = zrange / xrange

    w, h = 800, 600
    fig.update_layout(height=h, width=w)
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=(xmin, xmax),
                **addl_kwargs,
            ),
            yaxis=dict(
                range=(ymin, ymax),
                **addl_kwargs,
            ),
            zaxis=dict(
                range=(zmin, zmax),
                **addl_kwargs,
            ),
            aspectmode="manual",
            aspectratio=dict(x=xratio, y=yratio, z=zratio),
            camera=camera,
        )
    )
    return fig


def plot_hand_mesh(
    joint_angles,
    color="lightpink",
    opacity=1.0,
    flip=False,
    show_triangles=False,
    auto_range=False,
    clean_background=True,
    **mesh_kwargs,
):
    fig = _plot_hand_mesh_from_angles(
        joint_angles,
        color=color,
        opacity=opacity,
        flip=flip,
        show_triangles=show_triangles,
        **mesh_kwargs,
    )
    fig = _set_3d_plot_layout(
        fig, auto_range=auto_range, flip=flip, clean_background=clean_background
    )
    return fig


def generate_hand_mesh_frames_from_joint_angles(
    joint_angles, color="lightpink", flip=False, opacity=0.5, **mesh_kwargs
):
    frames = []
    initial_data = None
    for ind, ja in tqdm(enumerate(joint_angles), desc="Rendering"):

        hand_mesh = generate_hand_mesh_from_joint_angles(
            joint_angles=ja,
            color=color,
            flip=flip,
            opacity=opacity,
            **mesh_kwargs,
        )

        if initial_data is None:
            # First frame
            initial_data = [hand_mesh]

        frame = go.Frame(data=[hand_mesh], name=f"frame{ind}")
        frames.append(frame)

    return initial_data, frames


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


def animate_frames(
    initial_data,
    frames,
):
    """
    Creates an animation from frames.
    Creates the play button and sets the speed.
    """
    # See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Layout.html#plotly.graph_objects.Layout

    fig = go.Figure(
        data=initial_data,
        frames=frames,
        layout=go.Layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
        ),
    )

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]
    fig.update_layout(sliders=sliders)
    return fig


def get_plotly_animation_for_joint_angles(
    joint_angles,
    flip=False,
    title="",
    color="lightpink",
    opacity=1.0,
    auto_range=False,
    clean_background=True,
):
    initial, frames = generate_hand_mesh_frames_from_joint_angles(
        joint_angles, flip=flip, opacity=opacity, color=color
    )
    fig = animate_frames(initial, frames)
    fig = _set_3d_plot_layout(
        fig, auto_range=auto_range, flip=flip, clean_background=clean_background
    )

    return fig


def fig_to_array(fig, scale=1):
    # scale=2 to preserve perceived img quality
    fig_bytes = fig.to_image(format="png", scale=scale)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    frame_array = np.asarray(img)
    return frame_array


def _joint_angles_to_frame(joint_angles, **kwargs):
    fig = plot_hand_mesh(joint_angles=joint_angles, **kwargs)
    frame_array = fig_to_array(fig)
    return frame_array


def joint_angles_to_frames(joint_angles_t, **kwargs):
    frames = np.array(
        [
            _joint_angles_to_frame(joint_angles=joint_angles, **kwargs)
            for joint_angles in tqdm(joint_angles_t)
        ]
    )
    return frames


def joint_angles_to_frames_parallel(joint_angles, n_jobs=32, **kwargs):
    frames = np.array(
        ProgressParallel(n_jobs=n_jobs, total=len(joint_angles))(
            delayed(_joint_angles_to_frame)(ja, **kwargs) for ja in joint_angles
        )
    )
    return frames


def remove_alpha_channel(frames):
    return frames[..., :3]


def ik_failure_plot(session, ax=None):

    if ax is None:
        # Create a figure and axes with the specified figsize
        fig, ax = plt.subplots(figsize=(8, 2))

    ik_failure_mask = ~session.no_ik_failure.astype(bool)
    (line,) = ax.plot(session["joint_angles"].sum(-1), label="joint angle sum")

    # Iterate over the mask to find start and end indices of True regions
    start = None
    for i, failure in enumerate(ik_failure_mask):
        if failure and start is None:
            start = i
        elif not failure and start is not None:
            ax.axvspan(start, i, color="red", alpha=0.3)
            start = None

    # If the last region ends at the end of the array
    if start is not None:
        ax.axvspan(start, len(ik_failure_mask), color="red", alpha=0.3)

    ax.set_title(session.session_name + "\n" + session.metadata["stage"])

    # Create a custom legend entry for the red regions
    ik_failure_patch = Patch(color="red", alpha=0.3, label="IK failure")

    # Add both the line and the patch to the legend
    ax.legend(handles=[line, ik_failure_patch], bbox_to_anchor=(1, 1))

    return ax
