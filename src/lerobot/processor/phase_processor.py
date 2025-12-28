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


@ProcessorStepRegistry.register("dayuan/phase_processor")
@dataclass
class PhaseProcessorStep(ProcessorStep):
    """
    Processor Step for extracting phase from state to a independent dict to avoid normalized.

    Attributes:
        phase_num: The dimension of phase in the state.
    """
    phase_num: None | int = None

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
        if self.phase_num is None:
            return transition
        
        new_transition = transition.copy()
        obs = new_transition[TransitionKey.OBSERVATION]
        state = obs[OBS_STATE]
        phase = state[..., -1] # assuming phase is the last dimension
        obs["phase"] = phase        #  Batch, 1
        new_transition[TransitionKey.OBSERVATION] = obs

        return new_transition

    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns:
            The updated policy features dictionary.
        """
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns:
            A dictionary representation of the processor step configuration.
        """
        return {
            "phase_num": self.phase_num,
        }