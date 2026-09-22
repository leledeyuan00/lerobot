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

from lerobot.policies.residual_sac.configuration_residual_sac import ResidualSACConfig
from lerobot.policies.residual_sac.fact_backbone import FrozenFACT
from lerobot.policies.residual_sac.modeling_residual_sac import ResidualSACPolicy
from lerobot.policies.residual_sac.residual_action import IDENTITY_ROT6D, dataset_action_to_arms
from lerobot.policies.residual_sac.runner import ResidualPolicyRunner
from lerobot.utils.rotation_utils import random_rotations

# duplicated from conftest.py: see the note in test_fact_backbone.py (no importable package here). D is
# fact_feature_dim for the default pool_mode="structured" (4 slots: state, wrench, phase, visual) against
# the fixture's dim_model=32 -- NOT dim_model itself.
H, W = 64, 64
D = 4 * 32


def raw_obs():
    from lerobot.policies.residual_sac.residual_action import matrix_to_quat_xyzw

    state = torch.randn(27)
    state[3:7] = matrix_to_quat_xyzw(random_rotations(1))[0]
    state[16:20] = matrix_to_quat_xyzw(random_rotations(1))[0]
    state[26] = 1.0
    return {"observation.state": state, "observation.images.front": torch.rand(3, H, W)}


@pytest.fixture
def runner(fact_dir):
    path, _ = fact_dir
    fact = FrozenFACT.from_pretrained(str(path), device="cpu")
    torch.manual_seed(0)
    policy = ResidualSACPolicy(ResidualSACConfig(device="cpu", fact_feature_dim=D, push_to_hub=False))
    return ResidualPolicyRunner(fact, policy, deterministic=True)


def test_select_action_is_noop_at_init(runner):
    runner.reset()
    out = runner.select_action(raw_obs())
    assert out["action"].shape == (16,)
    # near-zero-init actor -> executed action ~= a_IL (position/gripper) with ~identity extra rotation
    assert torch.allclose(out["action"][:3], out["a_il"][:3], atol=1e-2)  # left pos untouched (fixed arm)
    assert torch.allclose(out["action"][8:11], out["a_il"][8:11], atol=5e-3)  # right pos ~ unchanged
    assert torch.allclose(out["residual"][4:], torch.tensor(IDENTITY_ROT6D), atol=0.1)


def test_select_action_returns_finite_outputs_over_an_episode(runner):
    runner.reset()
    for _ in range(5):
        out = runner.select_action(raw_obs())
        assert torch.isfinite(out["action"]).all()
        assert torch.isfinite(out["feature"]).all()
        assert out["feature"].shape == (D,)


def test_left_arm_passthrough_quaternion_untouched(runner):
    """Left (fixed) arm's orientation in the executed action must be exactly FACT's, not recomposed."""
    runner.reset()
    out = runner.select_action(raw_obs())
    assert torch.equal(out["action"][3:7], out["a_il"][3:7])


def test_encode_intervention_requires_prior_select_action(runner):
    with pytest.raises(RuntimeError):
        runner.encode_intervention(torch.zeros(16))


def test_encode_intervention_uses_last_a_il_and_inverse_composition(runner):
    runner.reset()
    runner.select_action(raw_obs())  # seeds _last_il_arms
    human = torch.zeros(16)
    human[0:3] = torch.randn(3) * 0.05
    human[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    human[8:11] = torch.randn(3) * 0.05
    human[11:15] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    label = runner.encode_intervention(human)
    assert label.shape == (10,)

    expected = runner.space.encode(runner._last_il_arms, dataset_action_to_arms(human[None]))[0]
    assert torch.allclose(label, expected, atol=1e-6)


def test_reset_clears_intervention_cache(runner):
    runner.reset()
    runner.select_action(raw_obs())
    assert runner._last_il_arms is not None
    runner.reset()
    assert runner._last_il_arms is None


def test_load_actor_state_dict_updates_in_place(runner):
    new_sd = {k: v + 1.0 for k, v in runner.policy.actor.state_dict().items()}
    runner.load_actor_state_dict(new_sd)
    for k, v in runner.policy.actor.state_dict().items():
        assert torch.equal(v, new_sd[k])


def test_from_pretrained_roundtrip(fact_dir, tmp_path):
    path, _ = fact_dir
    cfg = ResidualSACConfig(device="cpu", fact_feature_dim=D, push_to_hub=False)
    ResidualSACPolicy(cfg).save_pretrained(tmp_path)
    runner = ResidualPolicyRunner.from_pretrained(str(path), str(tmp_path), device="cpu")
    runner.reset()
    out = runner.select_action(raw_obs())
    assert out["action"].shape == (16,)
