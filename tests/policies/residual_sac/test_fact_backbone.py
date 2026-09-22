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

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.residual_sac.configuration_residual_sac import OBS_FACT_FEATURE, ResidualSACConfig
from lerobot.policies.residual_sac.fact_backbone import FrozenFACT, fact_structured_feature_dim
from lerobot.policies.residual_sac.modeling_residual_sac import ResidualSACPolicy
from lerobot.policies.residual_sac.residual_action import FACT_SELECTED_ACTION_INDICES, matrix_to_quat_xyzw
from lerobot.utils.rotation_utils import random_rotations

# NOTE: no `tests/policies/residual_sac/__init__.py` -> this package isn't import-able by relative path, so
# these are duplicated from `conftest.py` (which pytest picks up by auto-discovery, not import) rather than
# shared via a plain `from .conftest import ...`.
D, H, W, B = 32, 64, 64, 3
# The fixture's FACTConfig has robot_state_feature + wrench_dim + phase_num + one camera -> structured mode
# gives 4 slots (state, wrench, phase, visual): computed here, not hardcoded, so it tracks the fixture.
STRUCTURED_D = 4 * D


def raw_obs(b: int = B) -> dict[str, torch.Tensor]:
    state = torch.randn(b, 27)
    state[:, 3:7] = matrix_to_quat_xyzw(random_rotations(b))
    state[:, 16:20] = matrix_to_quat_xyzw(random_rotations(b))
    state[:, 26] = 1.0  # phase id
    return {"observation.state": state, "observation.images.front": torch.rand(b, 3, H, W)}


def test_fact_structured_feature_dim_matches_fixture(fact_dir):
    path, _ = fact_dir
    assert fact_structured_feature_dim(PreTrainedConfig.from_pretrained(path)) == STRUCTURED_D


def test_features_and_il_action(fact_dir):
    """Default pool_mode="structured"."""
    path, stats = fact_dir
    fact = FrozenFACT.from_pretrained(str(path), device="cpu")
    assert fact.feature_dim == STRUCTURED_D
    obs = raw_obs()
    out = fact(obs)
    assert out.feature.shape == (B, STRUCTURED_D) and not out.feature.requires_grad
    assert out.raw.shape == (B, 20)

    # a_IL = first chunk action; position / gripper un-normalised with the ACTION stats, rotation untouched
    batch = fact.preprocessor(obs)
    chunk = fact.policy.predict_action_chunk(batch)[:, 0]
    idx = FACT_SELECTED_ACTION_INDICES
    expected = chunk[:, :8] * stats["action"]["std"][idx] + stats["action"]["mean"][idx]
    assert torch.allclose(out.raw[:, :8], expected, atol=1e-5)
    assert torch.equal(out.raw[:, 8:], chunk[:, 8:])
    assert torch.allclose(out.arms["right"].pos, out.raw[:, 4:7])
    assert torch.allclose(out.arms["right"].rot @ out.arms["right"].rot.transpose(-1, -2),
                          torch.eye(3).expand(B, 3, 3), atol=1e-4)  # fmt: skip


def test_gap_pool_mode_is_mean_of_every_token_and_layer_selectable(fact_dir):
    path, _ = fact_dir
    obs = raw_obs()
    last = FrozenFACT.from_pretrained(str(path), device="cpu", pool_mode="gap")(obs).feature
    first = FrozenFACT.from_pretrained(str(path), device="cpu", feature_layer=0, pool_mode="gap")(obs).feature
    assert last.shape == (B, D) and not torch.allclose(last, first)

    fact = FrozenFACT.from_pretrained(str(path), device="cpu", pool_mode="gap")
    captured = []
    fact.policy.model.encoder.register_forward_hook(lambda m, i, o: captured.append(o))
    out = fact(obs)
    assert torch.allclose(out.feature, captured[-1].mean(dim=0), atol=1e-6)


def test_structured_pool_mode_keeps_state_and_wrench_as_their_own_untouched_slots(fact_dir):
    """The whole point of "structured": state/wrench must NOT be averaged away by ~hundreds of image
    tokens -- each is its own slot, exactly equal to the raw encoder token at that position."""
    path, _ = fact_dir
    fact = FrozenFACT.from_pretrained(str(path), device="cpu", pool_mode="structured")
    captured = []
    fact.policy.model.encoder.register_forward_hook(lambda m, i, o: captured.append(o))
    obs = raw_obs()
    out = fact(obs)

    tokens = captured[-1]  # (num_tokens, B, D): [latent, state, wrench, phase, *image_tokens]
    n_1d = fact.policy.model.n_1d_tokens
    assert n_1d == 4  # latent, state, wrench, phase
    # concatenation order (see FrozenFACT._pool_tokens): visual slot first, then the semantic tokens in
    # `_fact_token_layout`'s order (state, wrench, phase, env_state -- whichever are present)
    visual_slot, state_slot, wrench_slot, phase_slot = out.feature.split(D, dim=-1)

    assert torch.allclose(state_slot, tokens[1], atol=1e-6)
    assert torch.allclose(wrench_slot, tokens[2], atol=1e-6)
    assert torch.allclose(phase_slot, tokens[3], atol=1e-6)
    assert torch.allclose(visual_slot, tokens[n_1d:].mean(dim=0), atol=1e-6)
    # the drowning-out failure mode this exists to avoid: state/wrench must differ from a naive full-GAP
    naive_gap = tokens.mean(dim=0)
    assert not torch.allclose(state_slot, naive_gap, atol=1e-3)
    assert not torch.allclose(wrench_slot, naive_gap, atol=1e-3)


def test_structured_excludes_the_constant_latent_token(fact_dir):
    """At inference the VAE latent is always zeros -> the latent token is a fixed constant across calls;
    structured pooling must not waste a slot on it."""
    path, _ = fact_dir
    fact = FrozenFACT.from_pretrained(str(path), device="cpu", pool_mode="structured")
    f1 = fact(raw_obs()).feature
    f2 = fact(raw_obs()).feature  # different observation -> different features, but same fixed latent token
    assert f1.shape[-1] == STRUCTURED_D
    assert not torch.allclose(f1, f2)  # sanity: pooling is responsive to the actual observation


def test_phase_id_matches_the_phase_actually_fed_in(fact_dir):
    """The fixture's phase_num=4: phase_id must reflect whatever phase was in the raw obs, not a constant."""
    path, _ = fact_dir
    fact = FrozenFACT.from_pretrained(str(path), device="cpu")
    obs = raw_obs(b=5)
    obs["observation.state"][:, 26] = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.0])  # phase id is the last state dim
    out = fact(obs)
    assert out.phase_id is not None
    assert torch.equal(out.phase_id, torch.tensor([0, 1, 2, 3, 1]))


def test_phase_id_is_none_for_a_checkpoint_without_phase_conditioning(tmp_path_factory):
    """E.g. `cable_task2_single_wrench`, phase 2 trained in isolation with phase_num=None."""
    import numpy as np

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.fact.configuration_fact import FACTConfig
    from lerobot.policies.fact.modeling_fact import FACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(0)
    cfg = FACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (27,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, H, W)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (16,))},
        selected_state_shape=[18], selected_action_shape=[20], wrench_dim=12, phase_num=None,
        use_film=False,  # phase_num=None + use_film=True crashes in modeling_fact.py (pre-existing, unrelated
        # bug: `cond=encoder_phase_embed if use_film else None` needs `encoder_phase_embed`, only ever
        # assigned `if phase_num is not None`); real checkpoints already avoid this the same way, e.g.
        # cable_task2_single_wrench's config.json pairs phase_num=null with use_film=false.
        chunk_size=4, n_action_steps=1, dim_model=D, n_heads=2, dim_feedforward=64,
        n_encoder_layers=2, n_decoder_layers=1, latent_dim=8, n_vae_encoder_layers=1,
        pretrained_backbone_weights=None, device="cpu", push_to_hub=False,
    )  # fmt: skip
    stats = {
        "observation.state": {"mean": np.zeros(27, dtype=np.float32), "std": np.ones(27, dtype=np.float32)},
        "observation.images.front": {"mean": np.zeros((3, 1, 1), dtype=np.float32), "std": np.ones((3, 1, 1), dtype=np.float32)},
        "action": {"mean": np.zeros(16, dtype=np.float32), "std": np.ones(16, dtype=np.float32)},
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    path = tmp_path_factory.mktemp("fact_no_phase")
    FACTPolicy(cfg).save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)

    fact = FrozenFACT.from_pretrained(str(path), device="cpu")
    assert fact.feature_dim == 3 * D  # state + wrench + visual, no phase slot
    out = fact({"observation.state": torch.randn(3, 27), "observation.images.front": torch.rand(3, 3, H, W)})
    assert out.phase_id is None


def test_fact_stays_frozen_under_sac_training(fact_dir):
    path, _ = fact_dir
    fact = FrozenFACT.from_pretrained(str(path), device="cpu")
    assert not any(p.requires_grad for p in fact.parameters())
    fact.train()
    assert not fact.policy.training  # pinned to eval

    before = {k: v.clone() for k, v in fact.policy.state_dict().items()}
    sac = ResidualSACPolicy(ResidualSACConfig(device="cpu", fact_feature_dim=fact.feature_dim, push_to_hub=False))
    opts = torch.optim.Adam(sac.parameters(), lr=1e-2)
    for _ in range(3):
        out = fact(raw_obs())
        batch = {
            "action": torch.rand(B, 10) * 2 - 1,
            "reward": torch.ones(B),
            "state": {OBS_FACT_FEATURE: out.feature},
            "next_state": {OBS_FACT_FEATURE: out.feature},
            "done": torch.zeros(B),
        }
        loss = sac.forward(batch, "critic")["loss_critic"] + sac.forward(batch, "actor")["loss_actor"]
        opts.zero_grad()
        loss.backward()
        opts.step()
    assert all(p.grad is None for p in fact.parameters())
    after = fact.policy.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_select_action_matches_fact_policy_select_action(fact_dir):
    """Deployment path: single (unbatched) obs in, single un-ensembled-queue-consistent a_IL out."""
    path, stats = fact_dir
    fact = FrozenFACT.from_pretrained(str(path), device="cpu")
    fact.reset()
    obs = raw_obs(b=1)
    unbatched = {k: v[0] for k, v in obs.items()}

    out = fact.select_action(unbatched)
    assert out.feature.shape == (1, STRUCTURED_D) and out.raw.shape == (1, 20)

    idx = FACT_SELECTED_ACTION_INDICES
    batch = fact.preprocessor(unbatched)
    raw_action_20d = fact.policy.select_action(batch)
    expected = raw_action_20d[:, :8] * stats["action"]["std"][idx] + stats["action"]["mean"][idx]
    # NOTE: this calls policy.select_action a second time, which advances FACT's own action queue by one
    # more step; only the un-normalised decoding is checked here, not exact equality with `out.raw`.
    assert torch.allclose(expected[:, :3], out.raw[:, :3], atol=1e-2)


def test_select_action_works_without_reset_when_n_action_steps_is_one(fact_dir):
    """The fixture's `n_action_steps=1` means FACT re-infers (and the hook fires) on every call, so a fresh
    `FrozenFACT` produces a usable feature immediately -- `reset()` only matters between episodes."""
    path, _ = fact_dir
    fresh = FrozenFACT.from_pretrained(str(path), device="cpu")
    out = fresh.select_action({k: v[0] for k, v in raw_obs(b=1).items()})
    assert out.feature.shape == (1, STRUCTURED_D)


def test_reset_clears_the_cached_feature(fact_dir):
    path, _ = fact_dir
    fact = FrozenFACT.from_pretrained(str(path), device="cpu")
    fact.select_action({k: v[0] for k, v in raw_obs(b=1).items()})
    assert fact._last_feature is not None
    fact.reset()
    assert fact._last_feature is None


def test_invalid_pool_mode_rejected(fact_dir):
    path, _ = fact_dir
    import pytest

    with pytest.raises(ValueError):
        FrozenFACT.from_pretrained(str(path), device="cpu", pool_mode="mean")
