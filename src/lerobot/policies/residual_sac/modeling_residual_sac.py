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
"""Residual SAC head for a frozen FACT policy.

Reuse map w.r.t. `lerobot.policies.sac.modeling_sac`:
    reused verbatim : SACPolicy losses (critic / actor / temperature), target updates, `forward(model=...)`
                      interface used by `lerobot.rl.learner`, MLP, CriticHead, CriticEnsemble.
    replaced        : observation encoder (identity on the pooled FACT feature), the actor (identity-rotation
                      bias init, no-op start, 6D canonicalisation, a correct tanh-Normal).
    not present     : anything FACT. The FACT policy lives in `fact_backbone.py`, outside this nn.Module, so the
                      SAC optimisers can never touch it and the learner process never has to load it.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution

from lerobot.policies.residual_sac.configuration_residual_sac import OBS_FACT_FEATURE, ResidualSACConfig
from lerobot.policies.residual_sac.residual_action import IDENTITY_ROT6D, RES_ROT, canonicalize_rot6d
from lerobot.policies.sac.modeling_sac import MLP, SACPolicy


class FeatureEncoder(nn.Module):
    """Parameter-free "encoder": the observation IS the pooled FACT feature.

    It always detaches (actor, critic and target paths alike): there is nothing to train here, so a graph
    attached to the features could only ever lead back into FACT.
    """

    has_images = False

    def __init__(self, key: str, dim: int):
        super().__init__()
        self.key = key
        self._dim = dim

    def forward(self, obs: dict[str, Tensor], cache=None, detach: bool = False) -> Tensor:
        return obs[self.key].detach()

    @property
    def output_dim(self) -> int:
        return self._dim


class TanhNormal(TransformedDistribution):
    """Diagonal Normal(loc, scale) squashed by tanh.

    NOTE: `modeling_sac.TanhMultivariateNormalDiag` passes `scale_diag` as the *covariance* matrix of
    `MultivariateNormal`, i.e. the effective std is sqrt(scale). Here `scale` really is the std, which matters
    because the initial exploration noise of a residual policy has to be small and controllable.
    """

    def __init__(self, loc: Tensor, scale: Tensor):
        super().__init__(Independent(Normal(loc, scale), 1), [TanhTransform(cache_size=1)])


def canonicalize_action(action: Tensor) -> Tensor:
    """Replace the raw 6D rotation sub-vector of a residual action by its canonical (Gram-Schmidt) encoding."""
    return torch.cat([action[..., :RES_ROT.start], canonicalize_rot6d(action[..., RES_ROT])], dim=-1)


class ResidualActor(nn.Module):
    def __init__(
        self,
        encoder: FeatureEncoder,
        network: MLP,
        action_dim: int,
        std_min: float,
        std_max: float,
        init_std: float,
        init_final: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.log_std_min = math.log(std_min)
        self.log_std_max = math.log(std_max)

        out_features = next(m.out_features for m in reversed(network.net) if isinstance(m, nn.Linear))
        self.mean_layer = nn.Linear(out_features, action_dim)
        self.std_layer = nn.Linear(out_features, action_dim)
        for layer in (self.mean_layer, self.std_layer):
            nn.init.uniform_(layer.weight, -init_final, init_final)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            # pos / gripper: tanh(0) = 0 -> no-op. Rotation: the bias is the IDENTITY 6D encoding, not 0 (a
            # zero 6D vector is not a rotation), so that a near-zero weight update still gives R_res ~= I.
            self.mean_layer.bias[RES_ROT] = torch.tensor(IDENTITY_ROT6D)
            self.std_layer.bias.fill_(math.log(init_std))

    def forward(
        self, observations: dict[str, Tensor], observation_features=None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Features come from the frozen FACT: never let SAC gradients flow back into them.
        obs_enc = self.encoder(observations, cache=observation_features, detach=True)
        hidden = self.network(obs_enc)
        mean = self.mean_layer(hidden)
        std = self.std_layer(hidden).clamp(self.log_std_min, self.log_std_max).exp()

        dist = TanhNormal(mean, std)
        raw = dist.rsample()
        log_probs = dist.log_prob(raw)  # on the un-canonicalised sample: that is the density we sampled from
        return canonicalize_action(raw), log_probs, canonicalize_action(torch.tanh(mean))


class ResidualSACPolicy(SACPolicy):
    """SAC actor / twin critics over the residual action (10-d, or 9-d without a gripper residual, see
    `ResidualSACConfig.residual_include_gripper`), conditioned on pooled frozen-FACT features."""

    config_class = ResidualSACConfig
    name = "residual_sac"

    def __init__(self, config: ResidualSACConfig):
        super().__init__(config)

    def _init_encoders(self):
        self.shared_encoder = False
        self.encoder_critic = FeatureEncoder(OBS_FACT_FEATURE, self.config.fact_feature_dim)
        self.encoder_actor = self.encoder_critic

    def _init_actor(self, continuous_action_dim: int):
        kw = asdict(self.config.policy_kwargs)
        self.actor = ResidualActor(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            std_min=kw["std_min"],
            std_max=kw["std_max"],
            init_std=self.config.actor_init_std,
            init_final=kw["init_final"],
        )
        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            self.target_entropy = -np.prod(continuous_action_dim) / 2

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], deterministic: bool = False) -> Tensor:
        """Residual action (B, 10) for `batch = {OBS_FACT_FEATURE: (B, D)}`."""
        sampled, _, mean = self.actor(batch)
        return mean if deterministic else sampled
