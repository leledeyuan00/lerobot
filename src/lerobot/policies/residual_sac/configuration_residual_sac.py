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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.residual_sac.residual_action import ARMS, residual_action_dim
from lerobot.policies.sac.configuration_sac import PolicyConfig, SACConfig
from lerobot.utils.constants import ACTION

# The only observation the residual SAC sees: pooled encoder tokens of the frozen FACT policy (see
# fact_pool_mode below for exactly how they're pooled).
OBS_FACT_FEATURE = "observation.fact_feature"


@PreTrainedConfig.register_subclass("residual_sac")
@dataclass
class ResidualSACConfig(SACConfig):
    """Residual SAC on top of a frozen FACT policy.

    Reuses the whole `SACConfig` surface (so `lerobot.rl.learner` works unchanged), with these differences:
    * observations are a single pre-computed vector (`OBS_FACT_FEATURE`), no image / state encoders;
    * the action is the 10-d (or 9-d, see `residual_include_gripper`) residual of one arm (`residual_action.py`);
    * defaults tuned for short episodes and a terminal-only {0,1} reward.
    """

    # ---- frozen FACT foundation policy (only needed by the actor / demo converter, never by the learner)
    # Local dir (or hub id) of the FACT checkpoint: config.json + model.safetensors + policy_pre/postprocessor.
    fact_pretrained_path: str | None = None
    # How FrozenFACT pools the encoder's token sequence into the SAC state (see fact_backbone.py):
    # "structured" (default) -- one dim_model-wide slot per semantic 1D token (state, wrench, ...) kept
    #   separate, plus one more slot for all cameras' image tokens GAP-pooled together. Chosen over plain
    #   GAP because with ~hundreds of image tokens per camera, averaging them together with the single
    #   state/wrench tokens drowns the latter out -- exactly the signal contact-rich manipulation needs most.
    # "gap" -- mean-pool every token (state, wrench, every image token) into one dim_model vector. The
    #   original design; kept only as an A/B baseline against "structured", not recommended on its own.
    fact_pool_mode: str = "structured"
    # Size of FrozenFACT's pooled feature for this checkpoint + fact_pool_mode: dim_model for "gap";
    # dim_model * n_slots for "structured" (n_slots = 1 per present semantic token + 1 for images, e.g. 3 for
    # a checkpoint with state + wrench + at least one camera). `convert_demos.py` / `runner.py` assert this
    # matches `FrozenFACT.feature_dim` once the checkpoint is actually loaded and tell you the right value if
    # it doesn't, so treat a first-run mismatch error as "copy the number it gives you", not a bug to chase.
    fact_feature_dim: int = 512
    # Which FACT encoder layer's output is pooled. -1 = final encoder output (the current plan).
    # If the critic can't discriminate good/bad residuals, try an earlier layer (e.g. 2 of 4).
    fact_feature_layer: int = -1

    # ---- residual action space
    # Left arm is fixed (holds the connector); only the right arm runs the learned residual. The interface
    # (`ResidualActionSpace`, `ResidualPolicyRunner`) supports either arm / a future dual-arm extension, but
    # for this task only "right" is exercised.
    residual_arm: str = "right"
    # Residual rotation is composed as R_res @ R_IL ("world": twist expressed in the fixed world frame the
    # impedance controller commands in) or R_IL @ R_res ("local": twist about the end-effector's own axis).
    # Confirmed "world" for this setup. Mathematically the actor can express the same final rotations either
    # way (it's just which side the multiplication happens on) -- the choice matters for two things, not for
    # what's reachable: (a) it sets the coordinate frame the SAC actor's raw output lives in, so leave it
    # matched to the controller's frame rather than fighting an extra hidden rotation on every training step;
    # (b) the near-identity actor initialisation is only a true "start as a no-op, in the controller's own
    # frame" if the composition order matches how the controller will actually apply the correction.
    rotation_compose: str = "world"
    # Physical size of a residual of magnitude 1 (actor output is tanh-squashed to [-1, 1]), in the same
    # units as the dataset action. Pick these from `residual_stats.json` (written by `convert_demos.py`):
    # e.g. set `residual_pos_scale` to roughly the p99 of `demo_residual_pos_norm` so that the vast majority
    # of demo residuals land inside [-1, 1] (a demo residual that gets clipped can never be represented
    # exactly by the actor, and everything the actor emits is measured in multiples of this scale).
    residual_pos_scale: float = 0.01  # same unit as the dataset action position (m)
    residual_grip_scale: float = 0.01  # same unit as the dataset gripper command
    # Hard safety clamp on the executed residual rotation angle (rad), applied only when *executing* an
    # action (training sees the actor's raw, unclamped output) -- size it the same way, off
    # `demo_residual_rot_angle_rad` in `residual_stats.json`.
    max_residual_rot_rad: float = 0.1
    # Whether the gripper is part of the residual at all. False means the gripper always executes exactly
    # what the frozen FACT policy predicts -- no exploration/perturbation on it, no residual dimension for
    # it, and one fewer dimension for the actor/critic -- appropriate for a task where the gripper is fixed
    # (e.g. always closed) and there's nothing to learn there.
    residual_include_gripper: bool = True

    # ---- actor initialisation (no BC on the actor; it starts as a no-op residual)
    # Output-layer weights are drawn from U(-policy_kwargs.init_final, +init_final) ("near-zero weights").
    actor_init_std: float = 0.1  # initial exploration std in pre-tanh space (via the log-std bias)

    # ---- SAC defaults overridden for this problem
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    dataset_stats: dict[str, dict[str, list[float]]] | None = None
    policy_kwargs: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(use_tanh_squash=True, std_min=1e-3, std_max=0.5, init_final=1e-3)
    )
    discount: float = 0.95  # short single-attempt episodes: 0.9-0.95, not 0.99
    # Rewards are in {0, 1}; the SAC default alpha=1.0 would swamp them. Still auto-tuned (not hand-tuned).
    temperature_init: float = 0.01
    shared_encoder: bool = False  # no encoder parameters at all; features come pre-computed and detached
    use_torch_compile: bool = False  # keeps state_dict keys plain so warm-up checkpoints load cleanly
    online_step_before_learning: int = 200

    def __post_init__(self):
        super().__post_init__()
        if self.residual_arm not in ARMS:
            raise ValueError(f"residual_arm must be one of {ARMS}, got {self.residual_arm!r}")
        if self.rotation_compose not in ("local", "world"):
            raise ValueError(f"rotation_compose must be 'local' or 'world', got {self.rotation_compose!r}")
        if self.fact_pool_mode not in ("structured", "gap"):
            raise ValueError(f"fact_pool_mode must be 'structured' or 'gap', got {self.fact_pool_mode!r}")
        if self.num_discrete_actions is not None:
            raise ValueError("Gripper is a continuous residual here; num_discrete_actions must be None.")
        if not self.input_features:
            self.input_features = {
                OBS_FACT_FEATURE: PolicyFeature(type=FeatureType.STATE, shape=(self.fact_feature_dim,))
            }
        if not self.output_features:
            dim = residual_action_dim(self.residual_include_gripper)
            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(dim,))}

    def validate_features(self) -> None:
        if OBS_FACT_FEATURE not in self.input_features:
            raise ValueError(f"'{OBS_FACT_FEATURE}' must be in the input features.")
        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")
        expected = residual_action_dim(self.residual_include_gripper)
        if self.output_features[ACTION].shape[0] != expected:
            raise ValueError(
                f"The residual action must be {expected}-d (residual_include_gripper="
                f"{self.residual_include_gripper}), got {self.output_features[ACTION].shape[0]}."
            )

    @property
    def image_features(self) -> list[str]:
        return []
