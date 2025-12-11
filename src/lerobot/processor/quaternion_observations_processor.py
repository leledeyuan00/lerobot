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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_STATE, ACTION
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.rotation_utils import quaternion_to_matrix, matrix_to_rotation_6d


@ProcessorStepRegistry.register("dayuan/rotation_transform_processor")
@dataclass
class RotTransProcessorStep(ProcessorStep):
    """
    Processor step to convert quaternions to 6D representation.

    Attributes:
        features: A dictionary mapping feature groups to lists of feature names to be transformed.
                 example: {"left": ["ee_qx_l", "ee_qy_l", "ee_qz_l", "ee_qw_l"]
                           "right": ["ee_qx_r", "ee_qy_r", "ee_qz_r", "ee_qw_r"]}
        from_req: The source rotation representation. Currently only "quaternion" is supported.
        to_req: The target rotation representation. Currently only "rotation_6d" is supported
        float_dtype: The target floating-point dtype as a string (e.g., "float32", "float16", "bfloat16").
                     If None, the dtype is not changed.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Applies device and dtype conversion to all tensors in an environment transition.

        It iterates through the transition, finds all `torch.Tensor` objects (including those nested in
        dictionaries like `observation`), and processes them.

        Args:
            transition: The input `EnvTransition` object.

        Returns:
            A new `EnvTransition` object with all tensors moved to the target device and dtype.
        """
        new_transition = transition.copy()
        obs = new_transition[TransitionKey.OBSERVATION]
        state = obs[OBS_STATE]
        left_quat = state[..., 3:7] # xyzw
        left_quat = torch.cat([left_quat[..., 3:4], left_quat[..., 0:3]], dim=-1)  # to wxyz
        right_quat = state[..., 16:20] # xyzw
        right_quat = torch.cat([right_quat[..., 3:4], right_quat[..., 0:3]], dim=-1)  # to wxyz

        left_rot_6d = matrix_to_rotation_6d(quaternion_to_matrix(left_quat))
        right_rot_6d = matrix_to_rotation_6d(quaternion_to_matrix(right_quat))
        obs["observation_rotation"] = torch.cat([left_rot_6d, right_rot_6d], dim=-1)

        action = new_transition[TransitionKey.ACTION]
        if action is not None:
            left_quat_action = action[..., 3:7]
            left_quat_action = torch.cat([left_quat_action[..., 3:4], left_quat_action[..., 0:3]], dim=-1)  # to wxyz
            right_quat_action = action[..., 11:15]
            right_quat_action = torch.cat([right_quat_action[..., 3:4], right_quat_action[..., 0:3]], dim=-1)  # to wxyz
            
            left_rot_6d_action = matrix_to_rotation_6d(quaternion_to_matrix(left_quat_action))
            right_rot_6d_action = matrix_to_rotation_6d(quaternion_to_matrix(right_quat_action))
            obs["action_rotation"] = torch.cat([left_rot_6d_action, right_rot_6d_action], dim=-1)
            
        new_transition[TransitionKey.OBSERVATION] = obs
        # keep the original state and action
        # new_transition[OBS_STATE] = state
        # new_transition[ACTION] = action

        return new_transition

    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns:
            The updated policy features dictionary.
        """
        return features
