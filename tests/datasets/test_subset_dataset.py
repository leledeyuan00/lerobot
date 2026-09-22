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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.subset_dataset import SubsetStateActionDataset

STATE_NAMES = ["x", "y", "z", "gripper"]  # raw/live schema: 4-d, gripper interposed
ACTION_NAMES = ["x", "y", "z", "gripper"]
STATE_KEEP = ["x", "y", "z"]  # what the frozen checkpoint actually expects: 3-d, no gripper
ACTION_KEEP = ["x", "y", "z"]


@pytest.fixture
def ds(tmp_path):
    features = {
        "observation.state": {"dtype": "float32", "shape": (4,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (4,), "names": ACTION_NAMES},
    }
    d = LeRobotDataset.create("test/subset", fps=10, root=tmp_path / "ds", features=features, use_videos=False)
    for ep in range(2):
        for t in range(5):
            d.add_frame(
                {
                    "observation.state": torch.tensor([t, t + 0.5, t + 0.25, 1.0]),
                    "action": torch.tensor([t + 1.0, t + 1.5, t + 1.25, 1.0]),
                    "task": "probe",
                }
            )
        d.save_episode()
    d.finalize()
    return LeRobotDataset("test/subset", root=tmp_path / "ds")


def test_shapes_and_names_updated(ds):
    sub = SubsetStateActionDataset(ds, STATE_KEEP, ACTION_KEEP)
    assert sub.meta.features["observation.state"]["shape"] == [3]
    assert sub.meta.features["observation.state"]["names"] == STATE_KEEP
    assert sub.meta.features["action"]["shape"] == [3]
    assert sub.features is sub.meta.features  # convenience property matches LeRobotDataset.features


def test_getitem_drops_the_right_column(ds):
    sub = SubsetStateActionDataset(ds, STATE_KEEP, ACTION_KEEP)
    sample = sub[2]  # t=2 -> state [2, 2.5, 2.25, 1.0], action [3, 3.5, 3.25, 1.0]
    assert sample["observation.state"].shape == (3,)
    assert torch.allclose(sample["observation.state"], torch.tensor([2.0, 2.5, 2.25]))
    assert torch.allclose(sample["action"], torch.tensor([3.0, 3.5, 3.25]))


def test_index_resolution_is_order_and_subset_agnostic(ds):
    """Keep names in a different order / a strict subset -- indices must follow the requested order."""
    sub = SubsetStateActionDataset(ds, ["gripper", "x"], ["z"])
    sample = sub[0]
    assert torch.allclose(sample["observation.state"], torch.tensor([1.0, 0.0]))  # [gripper, x] at t=0
    assert torch.allclose(sample["action"], torch.tensor([1.25]))  # [z] at t=0 (action offset by +1)


def test_stats_are_sliced_consistently_with_features(ds):
    sub = SubsetStateActionDataset(ds, STATE_KEEP, ACTION_KEEP)
    stats = sub.meta.stats
    sliced_mean = torch.as_tensor(stats["observation.state"]["mean"])
    full_mean = torch.as_tensor(ds.meta.stats["observation.state"]["mean"])
    assert sliced_mean.shape == (3,)
    assert torch.allclose(sliced_mean, full_mean[:3])
    # scalar stats (e.g. count) must be left alone, not mis-sliced because they happen to be 1-d
    sliced_count = torch.as_tensor(stats["observation.state"]["count"])
    full_count = torch.as_tensor(ds.meta.stats["observation.state"]["count"])
    assert sliced_count.shape == full_count.shape


def test_len_and_dataloader_compatible(ds):
    sub = SubsetStateActionDataset(ds, STATE_KEEP, ACTION_KEEP)
    assert len(sub) == len(ds) == 10

    from torch.utils.data import DataLoader

    loader = DataLoader(sub, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["observation.state"].shape == (4, 3)
    assert batch["action"].shape == (4, 3)


def test_unknown_name_raises():
    class FakeMeta:
        features = {
            "observation.state": {"names": STATE_NAMES},
            "action": {"names": ACTION_NAMES},
        }
        stats = None

    class FakeBase:
        meta = FakeMeta()

    with pytest.raises(ValueError):
        SubsetStateActionDataset(FakeBase(), ["not_a_real_column"], ACTION_KEEP)
