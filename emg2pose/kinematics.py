# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from emg2pose.UmeTrack.lib.common.hand import HandModel
from emg2pose.UmeTrack.lib.common.hand_skinning import skin_landmarks

from torch import nn


DEFAULT_HAND_MODEL_FILE_REL = "./UmeTrack/dataset/generic_hand_model.json"


def load_hand_model_from_dict(hand_model_dict) -> HandModel:
    hand_tensor_dict = {}
    for k, v in hand_model_dict.items():
        hand_tensor_dict[k] = torch.tensor(v)

    hand_model = HandModel(**hand_tensor_dict)
    return hand_model


def load_hand_model_from_json(file_path: str) -> HandModel:
    with open(file_path, "r") as fp:
        hand_model_dict = json.load(fp)

    hand_model = load_hand_model_from_dict(hand_model_dict)

    return hand_model


class TorchNamedTuple(nn.Module):
    def __init__(self, named_tuple: NamedTuple | Any, requires_grad: bool = True):
        """Represent a NamedTuple as a nn.Module (collection of nn.Parameters)

        A note on the type signature: type checking for NamedTuples is pretty
        unintuitive. Even if arguments are a proper subtype of NamedTuple, they will
        still not be considered NamedTuples from a typing perspective.

        Here, we use Union[NamedTuple, Any] to avoid these linting issues at the source.

        Parameters
        ----------
        named_tuple : NamedTuple
            named tuple of torch.tensors
        requires_grad : bool, optional
            whether each nn.Parameter requires grad, by default True
        """
        super().__init__()

        for field_name in named_tuple._fields:
            field = getattr(named_tuple, field_name)
            setattr(self, field_name, nn.Parameter(field, requires_grad=requires_grad))

    @property
    def _fields(self) -> tuple[str, ...]:
        """Fields property to match the behavior of a traditional python named_tuple
        which is also a Tuple[str, ...] of the names within the named_tuple.

        Returns
        -------
        Tuple[str, ...]
            The names of the fields (here: parameters) of the TorchNamedTuple
        """
        return tuple(field_name for field_name, _ in self.named_parameters())


class TorchHandModel(TorchNamedTuple):
    def __init__(self, hand_model: HandModel):
        """A representation of HandModel that plays well with nn.Modules.

        Useful for automatically passing tensors in HandModel to different
        devices using `model.cuda()` or `model.cpu()`.

        Parameters
        ----------
        hand_model : HandModel
            hand model object
        """
        super().__init__(hand_model, requires_grad=False)

    def to_hand_model(self) -> HandModel:
        """Converts the nn.Parameter version (TorchHandModel) to HandModel.

        Returns
        -------
        HandModel
            hand_model with the values set from TorchHandModel
        """
        args = {}
        for param_name, param_val in self.named_parameters():
            args[param_name] = param_val
        return HandModel(**args)

    @property
    def device(self) -> torch.device:
        """Returns the device of this TorchHandModel."""
        parameters = next(iter(self._parameters.values()))
        assert parameters is not None, "No parameters found in TorchHandModel"
        return parameters.device


def load_default_hand_model() -> HandModel:
    """Load the default hand model.

    Returns
    -------
    HandModel
        default hand model for use in FK utilities
    """
    default_hand_model_file_path = os.path.join(
        os.path.dirname(__file__), DEFAULT_HAND_MODEL_FILE_REL
    )
    return load_hand_model_from_json(default_hand_model_file_path)


def apply_to_hand_model(
    hand_model: HandModel | TorchHandModel,
    field_fn: Callable[[torch.Tensor], torch.Tensor],
) -> HandModel:
    """Apply a function `field_fn` to each field in a HandModel
    and return a new HandModel option with the transformed fields contained.

    Useful for doing Tensor operations on the HandModel data object like
    broadcasting, unsqueezeing, indexing.

    Parameters
    ----------
    hand_model : Union[HandModel, TorchHandModel]
        hand_model to be transformed via field_fn
    field_fn : Callable[[torch.Tensor], torch.Tensor]
        function that ingests a torch.Tensor field and returns the updated
        torch.Tensor field

    Returns
    -------
    HandModel
        the transformed HandModel
    """
    args = {}
    for field_name in hand_model._fields:
        field_val = getattr(hand_model, field_name)
        field_val = field_fn(field_val)
        args[field_name] = field_val
    new_hand_model = HandModel(**args)

    return new_hand_model


def broadcast_hand_model_to(
    hand_model: HandModel | TorchHandModel, leading_dims: tuple[int, ...]
) -> HandModel:
    """Broadcast hand_model along leading dims.

    Useful for batching calls to `skin_landmarks` along a time and/or batch
    dimension.

    Examples
    --------

    add a batch dim of 64 to all fields::

        broadcast_hand_model_to(hand_model, (64,))

    add leading batch dims of (64, 400) to all fields::

        broadcast_hand_model_to(hand_model, (64, 400))

    Parameters
    ----------
    hand_model : Union[HandModel, TorchHandModel]
        hand_model to be expanded
    leading_dims : Tuple[int, ...]
        leading dims to be added to hand_model

    Returns
    -------
    HandModel
        broadcasted hand model
    """

    def broadcast_fn(field_val):
        return field_val.broadcast_to(*leading_dims, *field_val.shape)

    return apply_to_hand_model(hand_model, broadcast_fn)


def get_hand_model_leading_dims(
    hand_model: HandModel | TorchHandModel,
) -> tuple[int, ...]:
    """Utility function to get leading dims from HandModel

    Assumes fields are properly formatted, in which case the leading dims can
    be ascertained from the `joint_rotation_axes` field which normally has
    dim = 2. Any additional dims are batch / leading dims.

    Parameters
    ----------
    hand_model : Union[HandModel, TorchHandModel]
        hand model to get leading dims from

    Returns
    -------
    Tuple[int, ...]
        leading dims if any as a tuple of ints for the batch dims
    """
    leading_dims = hand_model.joint_rotation_axes.shape[:-2]
    return leading_dims  # type: ignore[no-any-return]


def get_joint_angle_leading_dims(joint_angles: torch.Tensor) -> tuple[int, ...]:
    """Utility function to get leading dims from joint angles.

    Assumes joint angles are normally of 1-D shape -- e.g. (22,). Any additional
    dims are deemed leading, batch dims.

    Parameters
    ----------
    joint_angles : torch.Tensor
        joint angles tensor to check leading dims

    Returns
    -------
    Tuple[int, ...]
        leading dims if any as a tuple of ints for the batch dim
    """
    leading_dims = joint_angles.shape[:-1]
    return leading_dims  # type: ignore[no-any-return]


def _broadcast_joint_angles_and_hand_model(
    joint_angles: torch.Tensor, hand_model: HandModel | TorchHandModel
) -> tuple[torch.Tensor, HandModel | TorchHandModel]:
    """Helper function to broadcast joint angles and hand models so that
    they work with `forward_kinematics`.

    Parameters
    ----------
    joint_angles : torch.Tensor
        joint angles of shape (..., 22)
    hand_model : Union[HandModel, TorchHandModel]
        hand model of shape (..., <hand_model_field_shapes>)

    Returns
    -------
    Tuple[torch.Tensor, Union[HandModel, TorchHandModel]]
        expanded joint angles and expanded hand model

    Raises
    ------
    ValueError
        when batch dim doesn't match in the Exception 1 case (see above)
    ValueError
        when leading dims dont match (and Exception 1 doesn't apply)
    ValueError
        when this function wasn't able to successfully match leading dims even
        after all the included logic (fail safe -- shouldn't trigger normally)
    """

    # get leading dims info
    joint_angle_leading_dims = get_joint_angle_leading_dims(joint_angles)
    hand_model_leading_dims = get_hand_model_leading_dims(hand_model)

    # if both have provided leading dims
    if joint_angle_leading_dims and hand_model_leading_dims:

        # SPECIAL CASE: joint angles has (B, T, 22) and hand_model has (B, ...)
        if len(joint_angle_leading_dims) == 2 and len(hand_model_leading_dims) == 1:
            batch_dim, time = joint_angle_leading_dims

            # make sure batch_dim (B) matches across joint_angles and hand_model
            if batch_dim != hand_model_leading_dims[0]:
                raise ValueError(
                    "if joint_angles and hand_model both have leading "
                    "dims, they **must** match in the batch dim. instead, got "
                    f"joint_angles={batch_dim} and "
                    f"hand_model={hand_model_leading_dims[0]}."
                )

            def _unsqueeze_and_expand_time(field):
                field = field.unsqueeze(1)

                # expand shape should be: (-1, time, ...)
                expand_shape = [-1 for _ in range(field.ndim)]
                expand_shape[1] = time

                field = field.expand(expand_shape)
                return field

            # unsqueeze hand_model time dimension (B, ...) --> (B, T, ...)
            hand_model = apply_to_hand_model(hand_model, _unsqueeze_and_expand_time)

        # ELSE IF: just make sure they are the same
        elif joint_angle_leading_dims != hand_model_leading_dims:
            raise ValueError(
                "got batched dims for both joint angles and hand_model but "
                "they did not match. got "
                f"joint_angles_leading_dims={joint_angle_leading_dims} and "
                f"hand_model_leading_dims={hand_model_leading_dims}."
            )

    # otherwise, update the other value (joint angles or hand_model) to match
    elif joint_angle_leading_dims:
        hand_model = broadcast_hand_model_to(hand_model, joint_angle_leading_dims)

    elif hand_model_leading_dims:
        joint_angles = joint_angles.expand(*hand_model_leading_dims, -1)

    # sanity check that leading dims now match
    new_joint_angle_leading_dims = get_joint_angle_leading_dims(joint_angles)
    new_hand_model_leading_dims = get_hand_model_leading_dims(hand_model)
    if new_joint_angle_leading_dims != new_hand_model_leading_dims:
        raise ValueError(
            "leading dims should match after the above logic. instead, got "
            f"joint_angles_leading_dims={new_joint_angle_leading_dims} and "
            f"hand_model_leading_dims={new_hand_model_leading_dims}."
        )

    return joint_angles, hand_model


def _batched_forward_kinematics(
    joint_angles: torch.Tensor,
    hand_model: HandModel | TorchHandModel | None = None,
    degrees: bool = False,
) -> torch.Tensor:
    """Get marker positions from joint angles (batching supported).

    Wrapper around UmeTrack `skin_landmarks` that respects batching.

    Parameters
    ----------
    joint_angles : torch.Tensor
        joint angles (in radians or degrees depending on `degrees` param).
        Batched tensors of shape (..., N) where typically N=22 are supported.
    hand_model : Union[HandModel, TorchHandModel], optional
        hand_model, by default None (uses default_hand_model).
    degrees: bool, optional
        whether the provided joint angles are in degrees. Defaults to False.

    Returns
    -------
    torch.Tensor
        landmark positions of shape (..., 21, 3)
    """

    if hand_model is None:
        hand_model = load_default_hand_model()

    if degrees:
        joint_angles = torch.deg2rad(joint_angles)

    # attempt to broadcast shapes such that they have same leading dims
    joint_angles, hand_model = _broadcast_joint_angles_and_hand_model(
        joint_angles, hand_model
    )

    # 4x4 affine identy transformation (unused in this code base)
    wrist_transform = torch.eye(4, dtype=joint_angles.dtype, device=joint_angles.device)

    # NOTE: manually broadcast an identity wrist_transform to be
    # compatible with UmeTrack.
    # if batch dim, expand wrist_transform to match joint_angles
    if joint_angles.ndim > 1:
        batch_dims = joint_angles.shape[:-1]
        wrist_transform = wrist_transform.reshape((1,) * len(batch_dims) + (4, 4))
        wrist_transform = wrist_transform.repeat(batch_dims + (1, 1))
    assert isinstance(hand_model, HandModel)
    return skin_landmarks(hand_model, joint_angles, wrist_transforms=wrist_transform)


def forward_kinematics(
    joint_angles: torch.Tensor, hand_model: HandModel | None = None
) -> torch.Tensor:
    """Convert joint angles to 3D coordinates."""

    # Add null wrist angles
    shape = [joint_angles.shape[0], 2, joint_angles.shape[2]]
    zeros = torch.zeros(*shape, dtype=joint_angles.dtype, device=joint_angles.device)
    joint_angles = torch.concat([joint_angles, zeros], 1)
    joint_angles = joint_angles.swapaxes(-2, -1)  # BCT -> BTC

    return _batched_forward_kinematics(joint_angles, hand_model)
