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
"""Wrap a `LeRobotDataset` and keep only a named subset of `observation.state` / `action` dimensions.

Ported (near-verbatim) from `iml_train_configs/subset_dataset.py`, the pattern already used to train the
`cable_task2_*` FACT checkpoints (see `iml_train_configs/custom_train.py`): those checkpoints were trained on
`STATE_KEEP_NAMES` / `ACTION_KEEP_NAMES` subsets of the raw, wider dataset schema (which also carries e.g.
per-arm gripper state and force/torque columns the model was never given). Any code -- FACT training,
residual-SAC demo conversion -- that reads the *live* dataset for one of these checkpoints needs to apply the
exact same subset, resolved once here by column name so it survives the raw schema growing further.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.utils.data as tud


class SubsetStateActionDataset(tud.Dataset):
    """Wrap a LeRobot dataset and keep only a subset of dims in `observation.state` / `action`.

    Also updates `.meta.features` and `.meta.stats` (sliced along with the data) so that anything reading
    dataset metadata (policy construction, normalizer stats) sees the correct, already-subsetted shapes.
    """

    def __init__(self, base_dataset, state_keep_names: list[str], action_keep_names: list[str]):
        super().__init__()
        self.base = base_dataset

        # 1) Read original names order from meta.features
        state_feat = self.base.meta.features["observation.state"]
        action_feat = self.base.meta.features["action"]

        state_names_full = state_feat["names"]
        action_names_full = action_feat["names"]

        # 2) Compute indices based on names
        self.state_indices = [state_names_full.index(n) for n in state_keep_names]
        self.action_indices = [action_names_full.index(n) for n in action_keep_names]

        # 3) Copy meta and update features & stats
        self.meta = copy.deepcopy(self.base.meta)

        # --- Update features ---
        self.meta.features["observation.state"]["names"] = list(state_keep_names)
        self.meta.features["observation.state"]["shape"] = [len(state_keep_names)]

        self.meta.features["action"]["names"] = list(action_keep_names)
        self.meta.features["action"]["shape"] = [len(action_keep_names)]

        # -- Update stats (if exists, usually mean/std/min/max etc.) ---
        if hasattr(self.meta, "stats") and self.meta.stats is not None:
            stats = self.meta.stats

            if "observation.state" in stats:
                full_dim_state = len(state_names_full)
                for k, v in list(stats["observation.state"].items()):
                    # Convert torch / np / list to np to check the last dimension length
                    if isinstance(v, torch.Tensor):
                        arr = v.cpu().numpy()
                        is_tensor = True
                    else:
                        arr = np.asarray(v)
                        is_tensor = False

                    # Only slice when the last dimension length == full_dim_state, otherwise (e.g., count=(1,)) keep original
                    if arr.ndim > 0 and arr.shape[-1] == full_dim_state:
                        arr = arr[..., self.state_indices]
                        if is_tensor:
                            stats["observation.state"][k] = torch.from_numpy(arr).to(v.dtype)
                        else:
                            stats["observation.state"][k] = arr
                    else:
                        # For example, count=(1,), keep original
                        stats["observation.state"][k] = v

            if "action" in stats:
                full_dim_action = len(action_names_full)
                for k, v in list(stats["action"].items()):
                    if isinstance(v, torch.Tensor):
                        arr = v.cpu().numpy()
                        is_tensor = True
                    else:
                        arr = np.asarray(v)
                        is_tensor = False

                    if arr.ndim > 0 and arr.shape[-1] == full_dim_action:
                        arr = arr[..., self.action_indices]
                        if is_tensor:
                            stats["action"][k] = torch.from_numpy(arr).to(v.dtype)
                        else:
                            stats["action"][k] = arr
                    else:
                        stats["action"][k] = v

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        if "observation.state" in sample:
            state = sample["observation.state"]
            sample["observation.state"] = state[..., self.state_indices]

        if "action" in sample:
            action = sample["action"]
            sample["action"] = action[..., self.action_indices]

        return sample

    @property
    def num_frames(self):
        return self.base.num_frames

    @property
    def num_episodes(self):
        return self.base.num_episodes

    @property
    def features(self):
        return self.meta.features
