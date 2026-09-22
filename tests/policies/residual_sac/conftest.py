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
"""Shared tiny-FACT-checkpoint fixture for the residual_sac test package."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.fact.configuration_fact import FACTConfig
from lerobot.policies.fact.modeling_fact import FACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.residual_sac.residual_action import matrix_to_quat_xyzw
from lerobot.utils.rotation_utils import random_rotations

FACT_D, H, W, B = 32, 64, 64, 3


def raw_obs(b: int = B) -> dict[str, torch.Tensor]:
    state = torch.randn(b, 27)
    state[:, 3:7] = matrix_to_quat_xyzw(random_rotations(b))
    state[:, 16:20] = matrix_to_quat_xyzw(random_rotations(b))
    state[:, 26] = 1.0  # phase id
    return {"observation.state": state, "observation.images.front": torch.rand(b, 3, H, W)}


@pytest.fixture(scope="package")
def fact_dir(tmp_path_factory):
    torch.manual_seed(0)
    cfg = FACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (27,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, H, W)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (16,))},
        selected_state_shape=[18], selected_action_shape=[20], wrench_dim=12, phase_num=4,
        chunk_size=4, n_action_steps=1, dim_model=FACT_D, n_heads=2, dim_feedforward=64,
        n_encoder_layers=2, n_decoder_layers=1, latent_dim=8, n_vae_encoder_layers=1,
        pretrained_backbone_weights=None, device="cpu", push_to_hub=False,
    )  # fmt: skip
    stats = {
        "observation.state": {"mean": torch.randn(27), "std": torch.rand(27) + 0.5},
        "observation.images.front": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        "action": {"mean": torch.randn(16), "std": torch.rand(16) + 0.5},
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    path = tmp_path_factory.mktemp("fact")
    FACTPolicy(cfg).save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    return path, stats
