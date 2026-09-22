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
"""Frozen FACT foundation policy used by the residual-SAC actor and the demo converter.

Two ways to query it, both returning a `FACTOutput` (pooled encoder feature + a_IL as per-arm `ArmAction`s,
un-normalised):
    * `forward(obs)`  -- training-time: always re-runs `predict_action_chunk` and takes its first action, no
      temporal ensembling. Used by `convert_demos.py` / `warmup_critic.py` (confirmed OK to differ from
      deployment: closed-loop re-inference at training time, ensembled smoothing only at deployment time).
    * `select_action(obs)` -- deployment-time: goes through `FACTPolicy.select_action` itself (its own action
      queue + `config.temporal_ensemble_coeff` smoothing), exactly like a normal FACT rollout. Used by
      `ResidualPolicyRunner` (`runner.py`).

Pooling (`pool_mode`, see `FrozenFACT.__init__`):
    * "structured" (default) -- one `dim_model`-wide slot per present *semantic* 1D encoder token (state,
      wrench, ...), kept separate, plus (if there are image inputs) one more slot that's the GAP of just the
      image tokens (all cameras pooled together). Concatenated into one `(B, n_slots * dim_model)` vector.
      This is the fix for a real failure mode of plain GAP: a handful of cameras contribute hundreds of
      tokens each, so averaging everything together lets the ~1 state token and ~1 wrench token get drowned
      out by sheer vision-token count -- exactly the wrench signal contact-rich manipulation most needs.
    * "gap" -- the original design: mean-pool *all* encoder tokens (state, wrench, every image token) into a
      single `(B, dim_model)` vector. Kept as an option to A/B against "structured", not because it's
      recommended -- it has the drowning-out problem above.
  Either way the (always informationless at inference -- see `_fact_token_layout`) latent token is excluded.

FACT is never trained here: parameters have `requires_grad=False`, the module is pinned to eval mode, both
entry points run under `torch.no_grad()`, and the returned feature is `.detach()`-ed and asserted grad-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.fact.configuration_fact import FACTConfig
from lerobot.policies.fact.modeling_fact import FACTPolicy
from lerobot.policies.residual_sac.residual_action import (
    FACT_OUT_DIM,
    FACT_SELECTED_ACTION_INDICES,
    ArmAction,
    fact_output_to_arms,
)
from lerobot.processor import UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION


class FACTActionCodec:
    """Un-normalises the 20-d FACT output.

    FACT regresses `[action[selected] (normalised with the dataset's ACTION stats), 6D rotations (raw)]`, so
    only the first 8 entries go through the (mean/std or min/max) un-normalisation; the rotations do not.
    """

    def __init__(self, stats: dict[str, Tensor], mode: NormalizationMode):
        idx = FACT_SELECTED_ACTION_INDICES
        self.mode = mode
        if mode == NormalizationMode.MEAN_STD:
            self.a, self.b = stats["mean"][idx], stats["std"][idx]  # x * std + mean
        elif mode == NormalizationMode.MIN_MAX:
            self.a, self.b = stats["min"][idx], stats["max"][idx]  # (x + 1) / 2 * (max - min) + min
        else:
            raise NotImplementedError(f"Action normalisation mode {mode} is not supported.")

    def decode(self, out: Tensor) -> Tensor:
        if out.shape[-1] != FACT_OUT_DIM:
            raise ValueError(f"Expected a {FACT_OUT_DIM}-d FACT output, got {tuple(out.shape)}")
        a, b = self.a.to(out), self.b.to(out)
        head, rot = out[..., :8], out[..., 8:]
        if self.mode == NormalizationMode.MEAN_STD:
            head = head * b + a
        else:
            head = (head + 1.0) / 2.0 * (b - a) + a
        return torch.cat([head, rot], dim=-1)

    @classmethod
    def from_postprocessor(cls, postprocessor) -> FACTActionCodec:
        step = next(s for s in postprocessor.steps if isinstance(s, UnnormalizerProcessorStep))
        mode = step.norm_map.get(FeatureType.ACTION, NormalizationMode.IDENTITY)
        return cls(step._tensor_stats[ACTION], mode)


@dataclass
class FACTOutput:
    feature: Tensor  # (B, feature_dim), detached -- see FrozenFACT.pool_mode for what feature_dim is
    arms: dict[str, ArmAction]  # a_IL, un-normalised
    raw: Tensor  # (B, 20) un-normalised (rotation still 6D)
    # (B,) long, the phase class index PhaseProcessorStep read off this observation -- None for a checkpoint
    # with `phase_num=None` (e.g. `cable_task2_single_wrench`, phase 2 in isolation). Exposed as its own
    # field (not just implicitly present as one of the slots inside `feature`) so that future work
    # conditioning the residual actor/critic on phase -- e.g. a FiLM layer or a cross-attention over phase,
    # mirroring FACT's own `use_film` -- has a clean signal to read without having to know which slice of
    # `feature` happens to be it. Deferred for now (see CLAUDE.md); this is just the plumbing for it.
    phase_id: Tensor | None = None


def _fact_token_layout(config: FACTConfig) -> dict[str, int]:
    """Index, within the encoder's per-step 1D-token block, of each present *semantic* token -- matching
    `FACT.forward()`'s exact append order: `[latent, state?, wrench?, phase?, image tokens...]`.

    Excludes the latent (always index 0): at inference (eval mode), `FACT`'s VAE latent is always the
    all-zeros sample (`FACT.forward`'s `else` branch: `use_vae and self.training` is never true in eval), so
    that token is a fixed constant carrying no information about the current observation -- nothing for
    structured pooling to gain by keeping it as its own slot.

    `phase` is kept as its own slot *on purpose*, not just incidentally: this project's FACT checkpoints may
    later be trained per-phase (multi-phase / phase-conditioned variants), and the residual policy needs to
    see the same phase signal FACT itself conditions on, for architectural consistency across the two. No
    `env_state` slot: none of this project's checkpoints have a (simulation-style) environment-state
    feature -- it's a real-robot, image-based setup throughout -- so it's dropped rather than carried as
    permanently-dead generality. (`FACTConfig.env_state_feature` still exists upstream; re-add a branch here
    if a future checkpoint ever actually uses it.)
    """
    idx = 1
    layout: dict[str, int] = {}
    if config.robot_state_feature:
        layout["state"] = idx
        idx += 1
    if config.wrench_dim is not None:
        layout["wrench"] = idx
        idx += 1
    if config.phase_num is not None:
        layout["phase"] = idx
        idx += 1
    return layout  # image tokens (if any) start at `idx`, which equals FACT.model.n_1d_tokens


def fact_structured_feature_dim(config: FACTConfig) -> int:
    """Width of `FrozenFACT`'s `pool_mode="structured"` feature for this FACT config: one `dim_model`-wide
    slot per present semantic 1D token, plus one more if there are image inputs (see `_fact_token_layout`)."""
    n_slots = len(_fact_token_layout(config)) + (1 if config.image_features else 0)
    return config.dim_model * n_slots


class FrozenFACT(nn.Module):
    def __init__(
        self,
        policy: FACTPolicy,
        preprocessor,
        postprocessor,
        feature_layer: int = -1,
        pool_mode: str = "structured",
    ):
        super().__init__()
        if policy.config.use_gru:
            raise NotImplementedError("use_gru FACT checkpoints need wrench-history keys; not supported here.")
        if pool_mode not in ("structured", "gap"):
            raise ValueError(f"pool_mode must be 'structured' or 'gap', got {pool_mode!r}")
        self.policy = policy.eval()
        self.preprocessor = preprocessor
        self.codec = FACTActionCodec.from_postprocessor(postprocessor)
        self.pool_mode = pool_mode
        self._token_layout = _fact_token_layout(policy.config)
        for p in self.policy.parameters():
            p.requires_grad_(False)

        encoder = self.policy.model.encoder
        target = encoder if feature_layer == -1 else encoder.layers[feature_layer]
        self._tokens: Tensor | None = None
        self._last_feature: Tensor | None = None  # select_action()'s cache, for queue-hit steps
        self._last_phase_id: Tensor | None = None  # kept in lockstep with _last_feature, same reason
        target.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output: Tensor) -> None:
        self._tokens = output  # (num_tokens, B, d_model)

    def train(self, mode: bool = True):  # FACT must stay in eval mode (dropout off, latent = 0)
        return super().train(False)

    def reset(self) -> None:
        """Call once per episode before the first `select_action`: clears FACT's action queue / ensembler."""
        self.policy.reset()
        self._last_feature = None
        self._last_phase_id = None

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cuda",
        feature_layer: int = -1,
        pool_mode: str = "structured",
    ) -> FrozenFACT:
        cfg = PreTrainedConfig.from_pretrained(path)
        if not isinstance(cfg, FACTConfig):
            raise TypeError(f"{path} is a '{cfg.type}' checkpoint, expected 'fact'.")
        cfg.device = device
        policy = FACTPolicy.from_pretrained(path, config=cfg).to(device)
        overrides: dict[str, Any] = {
            "device_processor": {"device": device},
            "normalizer_processor": {"device": device},
        }
        pre, post = make_pre_post_processors(cfg, pretrained_path=path, preprocessor_overrides=overrides)
        return cls(policy, pre, post, feature_layer=feature_layer, pool_mode=pool_mode)

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    @property
    def feature_dim(self) -> int:
        """What `FACTOutput.feature`'s last dimension actually is, for this checkpoint + `pool_mode`. Use
        this to set `ResidualSACConfig.fact_feature_dim` correctly -- `convert_demos.py` / `runner.py`
        assert the two match and tell you this value if they don't."""
        if self.pool_mode == "gap":
            return self.policy.config.dim_model
        return fact_structured_feature_dim(self.policy.config)

    def _pool_tokens(self) -> Tensor:
        if self._tokens is None:
            raise RuntimeError("FACT encoder hook did not fire.")
        tokens = self._tokens  # (num_tokens, B, d_model)
        if self.pool_mode == "gap":
            feature = tokens.mean(dim=0)
        else:
            parts = []
            if self.policy.config.image_features:
                n_1d = self.policy.model.n_1d_tokens
                parts.append(tokens[n_1d:].mean(dim=0))  # visual: GAP over every camera's tokens, one slot
            parts.extend(tokens[idx] for idx in self._token_layout.values())  # state, wrench, ... untouched
            feature = torch.cat(parts, dim=-1)  # (B, n_slots * d_model)
        feature = feature.detach().clone()
        self._tokens = None
        assert not feature.requires_grad, "FACT features must not carry gradients"
        return feature

    def _phase_id(self, batch: dict[str, Tensor]) -> Tensor | None:
        """(B,) long phase class index, as `PhaseProcessorStep` attaches it to the preprocessed batch under
        `"phase"` -- `None` when this checkpoint has no phase conditioning (`config.phase_num is None`)."""
        if self.policy.config.phase_num is None:
            return None
        return batch["phase"].flatten().long().detach().clone()

    @torch.no_grad()
    def forward(self, obs: dict[str, Any]) -> FACTOutput:
        """Training-time query. `obs`: raw observation dict/batch (dataset format; a batch dim is added if
        missing). Always a fresh `predict_action_chunk` call, first action of the chunk, no ensembling."""
        batch = self.preprocessor(obs)
        self._tokens = None
        chunk = self.policy.predict_action_chunk(batch)  # (B, chunk, 20), normalised
        feature = self._pool_tokens()
        raw = self.codec.decode(chunk[:, 0].float())
        return FACTOutput(feature=feature, arms=fact_output_to_arms(raw), raw=raw, phase_id=self._phase_id(batch))

    @torch.no_grad()
    def select_action(self, obs: dict[str, Any]) -> FACTOutput:
        """Deployment-time query, one call per control-loop tick. `obs`: a single (unbatched) raw
        observation dict; call `reset()` once at the start of each episode first.

        Delegates to `FACTPolicy.select_action`, so it reproduces FACT's own action queue / temporal
        ensembling (`config.temporal_ensemble_coeff`) exactly as a plain FACT rollout would. The encoder
        hook only fires when that internally re-runs `predict_action_chunk`; on a step where it just pops a
        queued action (only possible when `n_action_steps > 1`) the previous `feature` / `phase_id` / `arms`
        are reused, since the executed action came from that same chunk. With this task's `n_action_steps=1`
        the hook fires on every call, so this distinction doesn't currently arise in practice.
        """
        batch = self.preprocessor(obs)
        self._tokens = None
        action_20d = self.policy.select_action(batch)  # (B, 20), normalised, single already-picked action
        if self._tokens is not None:
            self._last_feature = self._pool_tokens()
            self._last_phase_id = self._phase_id(batch)
        if self._last_feature is None:
            raise RuntimeError("select_action() called before the encoder hook ever fired; call reset() first.")
        raw = self.codec.decode(action_20d.float())
        return FACTOutput(feature=self._last_feature, arms=fact_output_to_arms(raw), raw=raw, phase_id=self._last_phase_id)
