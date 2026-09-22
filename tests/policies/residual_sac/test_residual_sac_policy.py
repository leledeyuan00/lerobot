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

from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.residual_sac.configuration_residual_sac import OBS_FACT_FEATURE, ResidualSACConfig
from lerobot.policies.residual_sac.modeling_residual_sac import ResidualSACPolicy
from lerobot.policies.residual_sac.residual_action import IDENTITY_ROT6D
from lerobot.rl.buffer import ReplayBuffer

D, B = 32, 16


@pytest.fixture
def policy():
    torch.manual_seed(0)
    return ResidualSACPolicy(ResidualSACConfig(device="cpu", fact_feature_dim=D, push_to_hub=False))


def _batch(b=B):
    act = torch.rand(b, 10) * 2 - 1
    act[:, 4:] = torch.tensor(IDENTITY_ROT6D)
    return {
        "action": act,
        "reward": torch.zeros(b),
        "state": {OBS_FACT_FEATURE: torch.randn(b, D)},
        "next_state": {OBS_FACT_FEATURE: torch.randn(b, D)},
        "done": torch.zeros(b),
    }


def test_factory_registration_and_processor_precedence():
    cfg = make_policy_config("residual_sac", device="cpu")
    assert isinstance(cfg, ResidualSACConfig) and get_policy_class("residual_sac") is ResidualSACPolicy
    pre, _ = make_pre_post_processors(cfg)  # must NOT fall into the plain-SAC branch (subclass!)
    assert [type(s).__name__ for s in pre.steps] == ["DeviceProcessorStep"]


def test_actor_starts_as_noop_residual(policy):
    feats = {OBS_FACT_FEATURE: torch.randn(256, D)}
    _, _, mean = policy.actor(feats)  # deterministic action
    assert mean[:, :4].abs().max() < 0.05  # position / gripper ~ 0
    assert torch.allclose(mean[:, 4:], torch.tensor(IDENTITY_ROT6D).expand(256, 6), atol=0.05)


def test_actor_no_nan_on_degenerate_features(policy):
    a, lp, _ = policy.actor({OBS_FACT_FEATURE: torch.zeros(4, D)})
    assert torch.isfinite(a).all() and torch.isfinite(lp).all()


def test_actor_output_is_canonical_rotation(policy):
    a, _, _ = policy.actor({OBS_FACT_FEATURE: torch.randn(B, D)})
    r1, r2 = a[:, 4:7], a[:, 7:10]
    assert torch.allclose(r1.norm(dim=-1), torch.ones(B), atol=1e-5)
    assert torch.allclose((r1 * r2).sum(-1), torch.zeros(B), atol=1e-5)


def test_sac_losses_and_no_gradient_into_features(policy):
    batch = _batch()
    feats = batch["state"][OBS_FACT_FEATURE].requires_grad_(True)  # stand-in for "FACT output with a graph"
    losses = [policy.forward(batch, m) for m in ("critic", "actor", "temperature")]
    total = sum(next(iter(d.values())) for d in losses)
    total.backward()
    assert feats.grad is None, "SAC loss must never backprop into the FACT features"
    assert all(torch.isfinite(next(iter(d.values()))) for d in losses)
    assert all(p.grad is not None for p in policy.actor.parameters())


def test_expected_learner_interface(policy):
    for attr in ("actor", "critic_ensemble", "critic_target", "log_alpha", "temperature", "target_entropy"):
        assert hasattr(policy, attr)
    assert policy.config.vision_encoder_name is None  # learner.get_observation_features -> (None, None)
    groups = policy.get_optim_params()
    assert set(groups) == {"actor", "critic", "temperature"}
    policy.update_target_networks()
    policy.update_temperature()


def test_save_load_roundtrip(policy, tmp_path):
    policy.save_pretrained(tmp_path)
    loaded = ResidualSACPolicy.from_pretrained(tmp_path, config=policy.config)
    for (n, p), (_, q) in zip(policy.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert torch.equal(p, q), n


def test_replay_buffer_accepts_transitions(policy):
    buf = ReplayBuffer(capacity=32, device="cpu", state_keys=[OBS_FACT_FEATURE], use_drq=False)
    for i in range(20):
        buf.add(
            state={OBS_FACT_FEATURE: torch.randn(1, D)},
            action=torch.rand(1, 10),
            reward=float(i == 19),
            next_state={OBS_FACT_FEATURE: torch.randn(1, D)},
            done=i == 19,
            truncated=False,
        )
    b = buf.sample(8)
    loss = policy.forward(
        {k: b[k] for k in ("action", "reward", "state", "next_state", "done")}, "critic"
    )["loss_critic"]
    assert torch.isfinite(loss)


def test_gripper_excluded_gives_a_9d_action_space_throughout():
    """A task where the gripper is fixed (e.g. always closed): no gripper dimension anywhere in the
    actor/critic, not just clamped to zero -- one less thing for SAC to explore or spend capacity on."""
    torch.manual_seed(0)
    cfg = ResidualSACConfig(
        device="cpu", fact_feature_dim=D, push_to_hub=False, residual_include_gripper=False
    )
    assert cfg.output_features["action"].shape[0] == 9
    policy = ResidualSACPolicy(cfg)

    feats = {OBS_FACT_FEATURE: torch.randn(64, D)}
    sampled, log_probs, mean = policy.actor(feats)
    assert sampled.shape == (64, 9) and mean.shape == (64, 9)
    assert mean[:, :3].abs().max() < 0.05  # position ~ no-op at init
    assert torch.allclose(mean[:, 3:], torch.tensor(IDENTITY_ROT6D).expand(64, 6), atol=0.05)  # rot ~ identity

    batch = {
        "action": torch.cat([torch.rand(B, 3) * 2 - 1, torch.tensor(IDENTITY_ROT6D).expand(B, 6)], dim=-1),
        "reward": torch.zeros(B),
        "state": {OBS_FACT_FEATURE: torch.randn(B, D)},
        "next_state": {OBS_FACT_FEATURE: torch.randn(B, D)},
        "done": torch.zeros(B),
    }
    for model in ("critic", "actor", "temperature"):
        loss = next(iter(policy.forward(batch, model).values()))
        assert torch.isfinite(loss)
