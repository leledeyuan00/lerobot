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
"""`select_action`-style interface for an external (e.g. ROS2) control loop.

This repo ships no robot / ROS2 integration and no gym-style env: deployment happens outside lerobot, in
whatever process drives the impedance controller. `ResidualPolicyRunner` is the one thing that process needs
from us, mirroring `FACTPolicy.select_action`:

    runner = ResidualPolicyRunner.from_pretrained(fact_path, residual_path, device="cuda")
    runner.reset()                              # once per episode
    while ...:
        obs = ...                               # build the dataset-format observation dict from ROS2 topics
        out = runner.select_action(obs)
        publish_to_impedance_controller(out["action"])       # (16,), dataset action layout (quat xyzw)
        ...
        if human_took_over:
            residual_label = runner.encode_intervention(human_executed_action_16d)  # for HIL logging

For reporting transitions back to the learner for online SAC, see `lerobot.rl.residual.online_client`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.residual_sac.configuration_residual_sac import OBS_FACT_FEATURE
from lerobot.policies.residual_sac.fact_backbone import FrozenFACT
from lerobot.policies.residual_sac.modeling_residual_sac import ResidualSACPolicy
from lerobot.policies.residual_sac.residual_action import (
    ArmAction,
    ResidualActionSpace,
    arms_to_dataset_action,
    dataset_action_to_arms,
)


class ResidualPolicyRunner:
    def __init__(
        self,
        fact: FrozenFACT,
        policy: ResidualSACPolicy,
        space: ResidualActionSpace | None = None,
        deterministic: bool = True,
    ):
        """
        Args:
            fact: a `FrozenFACT` wrapping the frozen foundation policy.
            policy: the residual SAC policy (actor + critics; only the actor is used here).
            space: defaults to `ResidualActionSpace.from_config(policy.config)`.
            deterministic: use the actor's mean action (no sampling noise) -- the usual choice once online
                training has converged enough to deploy without exploration; set `False` to keep exploring.
        """
        self.fact = fact
        self.policy = policy.to(fact.device).eval()
        self.space = space or ResidualActionSpace.from_config(policy.config)
        self.deterministic = deterministic
        self._last_il_arms: dict[str, ArmAction] | None = None

    @classmethod
    def from_pretrained(
        cls,
        fact_path: str,
        residual_path: str,
        device: str = "cuda",
        feature_layer: int | None = None,
        deterministic: bool = True,
    ) -> ResidualPolicyRunner:
        cfg = PreTrainedConfig.from_pretrained(residual_path)
        policy = ResidualSACPolicy.from_pretrained(residual_path, config=cfg).to(device)
        fact = FrozenFACT.from_pretrained(
            fact_path,
            device=device,
            feature_layer=feature_layer if feature_layer is not None else cfg.fact_feature_layer,
            pool_mode=cfg.fact_pool_mode,
        )
        if fact.feature_dim != cfg.fact_feature_dim:
            raise ValueError(
                f"policy.fact_feature_dim={cfg.fact_feature_dim} doesn't match this FACT checkpoint's actual "
                f"pooled feature size for pool_mode={cfg.fact_pool_mode!r}: {fact.feature_dim}. This residual "
                f"policy was likely warmed up against a different FACT checkpoint or fact_pool_mode."
            )
        return cls(fact, policy, deterministic=deterministic)

    def reset(self) -> None:
        """Call once at the start of every episode (clears FACT's action queue / temporal ensembler)."""
        self.fact.reset()
        self._last_il_arms = None

    def load_actor_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Hot-swap the residual actor's weights, e.g. with parameters pushed by the learner."""
        self.policy.actor.load_state_dict(state_dict)

    @torch.no_grad()
    def select_action(self, observation: dict[str, Any]) -> dict[str, Tensor]:
        """`observation`: a single (unbatched) raw observation dict, dataset format (same as FACT's own
        `select_action`). Returns (all un-batched):
            "action"   : (16,) executed action = compose(a_IL, a_residual), dataset layout (quat xyzw) --
                         send this to the impedance controller.
            "a_il"     : (16,) FACT's own prediction, same layout, for logging / comparison.
            "residual" : (10,), or (9,) if `residual_include_gripper=False` -- the raw residual actor
                         output that produced "action".
            "feature"  : (d_model,) pooled FACT feature for this step -- this IS the SAC "state"; keep it if
                         you build training transitions yourself (see `lerobot.rl.residual.online_client` for
                         a ready-made helper that does this for you).
        """
        out = self.fact.select_action(observation)
        self._last_il_arms = out.arms

        state = {OBS_FACT_FEATURE: out.feature}
        residual = self.policy.select_action(state, deterministic=self.deterministic)
        final_arms = self.space.apply(out.arms, residual)

        return {
            "action": arms_to_dataset_action(final_arms)[0],
            "a_il": arms_to_dataset_action(out.arms)[0],
            "residual": residual[0],
            "feature": out.feature[0],
        }

    def encode_intervention(self, human_action: Tensor) -> Tensor:
        """Residual label for a human-executed 16-d dataset action, for HIL-intervention logging.

        Uses the a_IL of the *last* `select_action()` call (inverse composition, per CLAUDE.md -- never a
        plain subtraction of the two actions). Call `select_action()` at least once after `reset()` first.
        """
        if self._last_il_arms is None:
            raise RuntimeError("select_action() must be called at least once before encode_intervention().")
        human_arms = dataset_action_to_arms(human_action[None].to(self.fact.device).float())
        return self.space.encode(self._last_il_arms, human_arms)[0]
