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

import pytest
import torch

from lerobot.policies.residual_sac.residual_action import (
    ARMS,
    IDENTITY_ROT6D,
    RES_ROT,
    ArmAction,
    ResidualActionSpace,
    arms_to_dataset_action,
    canonicalize_rot6d,
    compose_rotation,
    dataset_action_to_arms,
    inverse_compose_rotation,
    limit_rotation_angle,
    matrix_to_quat_xyzw,
    quat_xyzw_to_matrix,
    residual_action_dim,
)
from lerobot.utils.rotation_utils import (
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    random_rotations,
    rotation_6d_to_matrix,
)

B = 32


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _arms(pos_scale=0.1):
    return {
        a: ArmAction(torch.randn(B, 3) * pos_scale, torch.randn(B, 1) * pos_scale, random_rotations(B))
        for a in ARMS
    }


@pytest.mark.parametrize("order", ["local", "world"])
def test_inverse_composition_is_exact(order):
    r_il, r_demo = random_rotations(B), random_rotations(B)
    r_res = inverse_compose_rotation(r_il, r_demo, order)
    assert torch.allclose(compose_rotation(r_il, r_res, order), r_demo, atol=1e-5)


def test_6d_subtraction_is_not_the_residual():
    """Guards the CLAUDE.md rule: 6D vectors must not be subtracted."""
    r_il, r_demo = random_rotations(B), random_rotations(B)
    naive = rotation_6d_to_matrix(matrix_to_rotation_6d(r_demo) - matrix_to_rotation_6d(r_il))
    assert not torch.allclose(naive, inverse_compose_rotation(r_il, r_demo), atol=1e-2)


def test_encode_apply_roundtrip_and_other_arm_untouched():
    space = ResidualActionSpace("right", pos_scale=0.5, grip_scale=0.5, max_rot_rad=4.0)
    il, tgt = _arms(), _arms()
    tgt["right"] = ArmAction(il["right"].pos + 0.1, il["right"].grip - 0.05, tgt["right"].rot)
    res = space.encode(il, tgt)
    assert res.shape == (B, 10) and res.abs().max() <= 1
    out = space.apply(il, res)
    assert torch.allclose(out["right"].pos, tgt["right"].pos, atol=1e-5)
    assert torch.allclose(out["right"].grip, tgt["right"].grip, atol=1e-5)
    assert torch.allclose(out["right"].rot, tgt["right"].rot, atol=1e-5)
    assert torch.equal(out["left"].pos, il["left"].pos) and torch.equal(out["left"].rot, il["left"].rot)


def test_identity_residual_is_noop():
    space = ResidualActionSpace("left")
    il = _arms()
    res = torch.zeros(B, 10)
    res[:, 4:] = torch.tensor(IDENTITY_ROT6D)
    out = space.apply(il, res)["left"]
    assert torch.allclose(out.pos, il["left"].pos) and torch.allclose(out.rot, il["left"].rot, atol=1e-6)


def test_canonicalize_is_encoding_invariant_and_zero_safe():
    a = torch.tensor([[0.76, 0, 0, 0, 0.76, 0]])
    assert torch.allclose(canonicalize_rot6d(a), torch.tensor([IDENTITY_ROT6D]), atol=1e-6)
    assert torch.isfinite(canonicalize_rot6d(torch.zeros(1, 6))).all()  # degenerate but never NaN


def test_rotation_limit():
    big = random_rotations(B)
    assert matrix_to_axis_angle(limit_rotation_angle(big, 0.1)).norm(dim=-1).max() <= 0.1 + 1e-5
    small = rotation_6d_to_matrix(torch.tensor([IDENTITY_ROT6D])).repeat(B, 1, 1)
    assert torch.allclose(limit_rotation_angle(small, 0.1), small, atol=1e-6)


def test_dataset_action_layout_roundtrip():
    arms = _arms()
    a16 = arms_to_dataset_action(arms)
    assert a16.shape == (B, 16)
    back = dataset_action_to_arms(a16)
    for arm in ARMS:
        assert torch.allclose(back[arm].rot, arms[arm].rot, atol=1e-5)
        assert torch.allclose(back[arm].pos, arms[arm].pos)
    # gripper indices 7 / 15, quaternion slots 3:7 / 11:15 (xyzw)
    assert torch.allclose(a16[:, 7:8], arms["left"].grip) and torch.allclose(a16[:, 15:16], arms["right"].grip)
    assert torch.allclose(quat_xyzw_to_matrix(a16[:, 11:15]), arms["right"].rot, atol=1e-5)
    assert (matrix_to_quat_xyzw(arms["left"].rot)[:, 3] >= 0).all()


def test_residual_action_dim():
    assert residual_action_dim(include_gripper=True) == 10
    assert residual_action_dim(include_gripper=False) == 9


def test_gripper_excluded_shrinks_the_residual_and_never_touches_the_gripper():
    space = ResidualActionSpace("right", pos_scale=0.5, grip_scale=0.5, max_rot_rad=4.0, include_gripper=False)
    assert space.action_dim == 9
    il, tgt = _arms(), _arms()
    # a large gripper difference in the demo target -- must be silently ignored, not clamped-to-±1-and-used
    tgt["right"] = ArmAction(tgt["right"].pos, il["right"].grip + 10.0, tgt["right"].rot)

    label = space.encode(il, tgt)
    assert label.shape == (B, 9)
    # rotation is still the last 6 columns of the now-9-d label (no grip column in between anymore)
    assert torch.allclose(label[..., RES_ROT], label[..., 3:9])

    out = space.apply(il, label)["right"]
    assert torch.equal(out.grip, il["right"].grip)  # untouched: no residual dimension carries it


def test_gripper_included_vs_excluded_same_pos_rot_roundtrip():
    """Excluding the gripper must not perturb the pos/rot part of encode()/apply()."""
    il, tgt = _arms(), _arms()
    with_grip = ResidualActionSpace("right", pos_scale=0.3, grip_scale=0.3, max_rot_rad=4.0, include_gripper=True)
    no_grip = ResidualActionSpace("right", pos_scale=0.3, grip_scale=0.3, max_rot_rad=4.0, include_gripper=False)

    label_with = with_grip.encode(il, tgt)
    label_without = no_grip.encode(il, tgt)
    assert torch.allclose(label_with[..., :3], label_without[..., :3])  # pos
    assert torch.allclose(label_with[..., RES_ROT], label_without[..., RES_ROT])  # rot

    out_with = with_grip.apply(il, label_with)["right"]
    out_without = no_grip.apply(il, label_without)["right"]
    assert torch.allclose(out_with.pos, out_without.pos, atol=1e-5)
    assert torch.allclose(out_with.rot, out_without.rot, atol=1e-5)
