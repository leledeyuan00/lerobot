#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Residual action space on top of the (frozen) FACT policy.

Position / gripper are plain vector spaces (add / subtract). The 6D rotation representation is NOT: it is
composed on SO(3) via rotation matrices (see CLAUDE.md, "6D rotation composition").

Layouts (inferred from `FACTPolicy.forward` / `RotTransProcessorStep`):
    * executed / dataset action, 16-d:
        [l_pos(3), l_quat_xyzw(4), l_grip(1), r_pos(3), r_quat_xyzw(4), r_grip(1)]
    * FACT network output, 20-d (position/gripper normalised like `action`, rotation raw 6D):
        [l_pos(3), l_grip(1), r_pos(3), r_grip(1), l_rot6d(6), r_rot6d(6)]
    * residual action of ONE arm, 10-d, or 9-d if the gripper isn't part of the residual (this is what SAC's
      actor/critic see, all entries in [-1, 1]):
        [d_pos(3) / pos_scale, (d_grip(1) / grip_scale,) residual_rot6d(6)]
      `residual_rot6d` is the *canonical* 6D encoding (first two rows of a rotation matrix) of `R_res`,
      identity = [1, 0, 0, 0, 1, 0]. See `ResidualActionSpace(include_gripper=...)` / `ResidualSACConfig.
      residual_include_gripper` -- when a task's gripper is fixed (e.g. always closed) it's dropped from the
      residual entirely (the gripper still executes, just always exactly the frozen FACT policy's own
      prediction), rather than kept as a residual dimension the actor has to learn to leave near zero.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from lerobot.utils.rotation_utils import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    standardize_quaternion,
)

ARMS = ("left", "right")

# 16-d executed / dataset action
ACTION_DIM = 16
ACTION_POS = {"left": slice(0, 3), "right": slice(8, 11)}
ACTION_QUAT = {"left": slice(3, 7), "right": slice(11, 15)}  # xyzw
ACTION_GRIP = {"left": 7, "right": 15}

# FACT output, 20-d
FACT_OUT_DIM = 20
FACT_SELECTED_ACTION_INDICES = [0, 1, 2, 7, 8, 9, 10, 15]  # same list as FACTPolicy.forward
FACT_OUT_POS = {"left": slice(0, 3), "right": slice(4, 7)}
FACT_OUT_GRIP = {"left": 3, "right": 7}
FACT_OUT_ROT = {"left": slice(8, 14), "right": slice(14, 20)}

# residual action of one arm: [pos3, (grip1), rot6] -- rot6 is always the *last* 6 entries, whether or not
# the gripper residual is included (see `ResidualActionSpace.include_gripper`), so `RES_POS`/`RES_ROT` are
# fixed/negative-indexed and work for either the 9-d (no gripper) or 10-d (with gripper) layout unchanged;
# only `RES_GRIP` (meaningful/used only when the gripper residual is included) is a fixed forward index.
RES_POS = slice(0, 3)
RES_GRIP = slice(3, 4)
RES_ROT = slice(-6, None)
RESIDUAL_DIM = 10  # with gripper; `residual_action_dim(include_gripper=False)` is 9
IDENTITY_ROT6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def residual_action_dim(include_gripper: bool) -> int:
    return RESIDUAL_DIM if include_gripper else RESIDUAL_DIM - 1


class ArmAction(NamedTuple):
    """One arm's action in a vector-space-friendly form: rotation is a matrix, not a 6D vector."""

    pos: Tensor  # (B, 3)
    grip: Tensor  # (B, 1)
    rot: Tensor  # (B, 3, 3)


# --------------------------------------------------------------------------------------------------
# rotation helpers
# --------------------------------------------------------------------------------------------------
def canonicalize_rot6d(d6: Tensor) -> Tensor:
    """Gram-Schmidt an arbitrary 6D vector and re-encode it as the first two rows of the resulting matrix.

    Two 6D vectors describing the same rotation (e.g. `[1,0,0,0,1,0]` and `[.76,0,0,0,.76,0]`) map to the
    same canonical vector, so the critic only ever sees one encoding per rotation. Differentiable.
    """
    return matrix_to_rotation_6d(rotation_6d_to_matrix(d6))


def compose_rotation(r_il: Tensor, r_res: Tensor, order: str = "local") -> Tensor:
    """`local`: R_IL @ R_res (residual expressed in the IL pose's own frame); `world`: R_res @ R_IL."""
    if order == "local":
        return r_il @ r_res
    if order == "world":
        return r_res @ r_il
    raise ValueError(f"rotation compose order must be 'local' or 'world', got {order!r}")


def inverse_compose_rotation(r_il: Tensor, r_target: Tensor, order: str = "local") -> Tensor:
    """Residual rotation such that `compose_rotation(r_il, R_res, order) == r_target`."""
    if order == "local":
        return r_il.transpose(-1, -2) @ r_target
    if order == "world":
        return r_target @ r_il.transpose(-1, -2)
    raise ValueError(f"rotation compose order must be 'local' or 'world', got {order!r}")


@torch.no_grad()
def limit_rotation_angle(rot: Tensor, max_angle: float) -> Tensor:
    """Clamp the rotation angle of `rot` (B,3,3) to `max_angle` radians, keeping its axis. No gradient."""
    if max_angle <= 0:
        return rot
    aa = matrix_to_axis_angle(rot)
    angle = aa.norm(dim=-1, keepdim=True)
    scale = torch.clamp(max_angle / angle.clamp_min(1e-9), max=1.0)
    return axis_angle_to_matrix(aa * scale)


def quat_xyzw_to_matrix(quat_xyzw: Tensor) -> Tensor:
    quat_wxyz = torch.cat([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], dim=-1)
    return quaternion_to_matrix(quat_wxyz)


def matrix_to_quat_xyzw(rot: Tensor) -> Tensor:
    quat_wxyz = standardize_quaternion(matrix_to_quaternion(rot))  # w >= 0
    return torch.cat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)


# --------------------------------------------------------------------------------------------------
# layout conversions
# --------------------------------------------------------------------------------------------------
def dataset_action_to_arms(action: Tensor) -> dict[str, ArmAction]:
    """(B, 16) raw (un-normalised) executed/dataset action -> per-arm `ArmAction`."""
    return {
        arm: ArmAction(
            pos=action[..., ACTION_POS[arm]],
            grip=action[..., ACTION_GRIP[arm] : ACTION_GRIP[arm] + 1],
            rot=quat_xyzw_to_matrix(action[..., ACTION_QUAT[arm]]),
        )
        for arm in ARMS
    }


def arms_to_dataset_action(arms: dict[str, ArmAction]) -> Tensor:
    """Per-arm `ArmAction` -> (B, 16) action in the dataset / impedance-controller layout (quat xyzw)."""
    batch = arms["left"].pos.shape[:-1]
    out = arms["left"].pos.new_zeros(*batch, ACTION_DIM)
    for arm in ARMS:
        out[..., ACTION_POS[arm]] = arms[arm].pos
        out[..., ACTION_QUAT[arm]] = matrix_to_quat_xyzw(arms[arm].rot)
        out[..., ACTION_GRIP[arm] : ACTION_GRIP[arm] + 1] = arms[arm].grip
    return out


def fact_output_to_arms(out: Tensor) -> dict[str, ArmAction]:
    """(B, 20) FACT output with position/gripper ALREADY un-normalised -> per-arm `ArmAction`."""
    return {
        arm: ArmAction(
            pos=out[..., FACT_OUT_POS[arm]],
            grip=out[..., FACT_OUT_GRIP[arm] : FACT_OUT_GRIP[arm] + 1],
            rot=rotation_6d_to_matrix(out[..., FACT_OUT_ROT[arm]]),
        )
        for arm in ARMS
    }


# --------------------------------------------------------------------------------------------------
# residual action space
# --------------------------------------------------------------------------------------------------
class ResidualActionSpace:
    """Maps between (a_IL, a_target) and the normalised residual of the arm that SAC controls: 10-d
    `[pos3, grip1, rot6]`, or 9-d `[pos3, rot6]` when `include_gripper=False` -- rot6 is always the last 6
    entries either way (see `RES_ROT`).

    * `encode(a_il, a_target)`: demo / human action -> residual label (inverse composition, used for the
      demo warm-up buffer and for human interventions).
    * `apply(a_il, residual)`: IL action + residual -> executed action (forward composition).
    """

    def __init__(
        self,
        arm: str = "right",
        pos_scale: float = 0.01,
        grip_scale: float = 0.01,
        rotation_compose: str = "local",
        max_rot_rad: float = 0.1,
        include_gripper: bool = True,
    ):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        self.arm = arm
        self.pos_scale = pos_scale
        self.grip_scale = grip_scale
        self.rotation_compose = rotation_compose
        self.max_rot_rad = max_rot_rad
        self.include_gripper = include_gripper
        self.action_dim = residual_action_dim(include_gripper)

    @classmethod
    def from_config(cls, cfg) -> ResidualActionSpace:
        return cls(
            arm=cfg.residual_arm,
            pos_scale=cfg.residual_pos_scale,
            grip_scale=cfg.residual_grip_scale,
            rotation_compose=cfg.rotation_compose,
            max_rot_rad=cfg.max_residual_rot_rad,
            include_gripper=cfg.residual_include_gripper,
        )

    def encode(self, a_il: dict[str, ArmAction], a_target: dict[str, ArmAction]) -> Tensor:
        """Inverse composition. Position/gripper: plain subtraction. Rotation: R_IL^-1 @ R_target.

        When `include_gripper` is False, the gripper is dropped entirely (not just clamped to 0): the label
        never encodes a gripper delta, matching `apply()` leaving the gripper untouched at execution time.
        """
        il, tgt = a_il[self.arm], a_target[self.arm]
        d_pos = ((tgt.pos - il.pos) / self.pos_scale).clamp(-1.0, 1.0)
        r_res = inverse_compose_rotation(il.rot, tgt.rot, self.rotation_compose)
        parts = [d_pos]
        if self.include_gripper:
            parts.append(((tgt.grip - il.grip) / self.grip_scale).clamp(-1.0, 1.0))
        parts.append(matrix_to_rotation_6d(r_res))
        return torch.cat(parts, dim=-1)

    def apply(
        self, a_il: dict[str, ArmAction], residual: Tensor, limit_rotation: bool = True
    ) -> dict[str, ArmAction]:
        """Forward composition. Returns both arms; the non-residual arm is passed through untouched, and so
        is the residual arm's gripper when `include_gripper` is False (no residual to apply)."""
        il = a_il[self.arm]
        r_res = rotation_6d_to_matrix(residual[..., RES_ROT])
        if limit_rotation:
            r_res = limit_rotation_angle(r_res, self.max_rot_rad)
        grip = il.grip + self.grip_scale * residual[..., RES_GRIP] if self.include_gripper else il.grip
        final = ArmAction(
            pos=il.pos + self.pos_scale * residual[..., RES_POS],
            grip=grip,
            rot=compose_rotation(il.rot, r_res, self.rotation_compose),
        )
        return {**a_il, self.arm: final}

    def rotation_clipped_fraction(self, residual: Tensor) -> Tensor:
        """Fraction of residuals whose rotation exceeds `max_rot_rad` (diagnostic for the exec-time clamp)."""
        angle = matrix_to_axis_angle(rotation_6d_to_matrix(residual[..., RES_ROT])).norm(dim=-1)
        return (angle > self.max_rot_rad).float().mean()
