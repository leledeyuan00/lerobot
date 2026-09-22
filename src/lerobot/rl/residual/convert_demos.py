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
"""Warm-up step 1: teleoperated demos (s, a_demo) -> residual transitions (s, a_residual, r, s') dataset.

For every demo frame the frozen FACT policy gives a_IL(s) and the pooled feature. The residual label is
    position / gripper : (a_demo - a_IL) / scale            (plain subtraction)
    rotation           : 6D( R_IL^-1 @ R_demo )              (inverse composition; NOT 6D subtraction)
Reward is terminal-only. The result is a LeRobotDataset that `lerobot.rl.learner` consumes as its offline
buffer (`dataset.repo_id`/`dataset.root`) and that `warmup_critic.py` pretrains the critics on.

    python -m lerobot.rl.residual.convert_demos \
        --policy.type=residual_sac --policy.fact_pretrained_path=/path/to/fact/pretrained_model \
        --policy.residual_arm=right --source_repo_id=leledeyuan/mating_cable_3cam \
        --out_repo_id=leledeyuan/mating_cable_residual --out_root=/data/mating_cable_residual --phases='[1]'

If the frozen FACT checkpoint was trained on a *subset* of the live dataset's `observation.state` / `action`
columns (as `cable_task2_*` was -- see `iml_train_configs/{custom_train,subset_dataset}.py`: the live dataset
carries e.g. `gripper_l`/`gripper_r` state and per-arm force/torque that checkpoint never saw), pass
`--state_keep_names=[...]` / `--action_keep_names=[...]` (the exact column names, in order, the checkpoint
expects -- copy them from whatever `STATE_KEEP_NAMES`/`ACTION_KEEP_NAMES` that checkpoint was trained with).
Feeding the live schema straight to a checkpoint expecting fewer columns fails loudly (a normalizer shape
mismatch), it does not silently produce wrong numbers -- but get this wrong the *other* way (right column
count, wrong column identities) and it will silently produce wrong numbers, so double check against the
checkpoint's own training config before running for real.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.subset_dataset import SubsetStateActionDataset
from lerobot.policies.residual_sac.configuration_residual_sac import OBS_FACT_FEATURE
from lerobot.policies.residual_sac.fact_backbone import FrozenFACT
from lerobot.policies.residual_sac.residual_action import (
    ResidualActionSpace,
    dataset_action_to_arms,
    inverse_compose_rotation,
)
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION, DONE, REWARD
from lerobot.utils.rotation_utils import matrix_to_axis_angle
from lerobot.utils.utils import init_logging


@dataclass
class ConvertDemosConfig:
    policy: PreTrainedConfig | None = None  # type=residual_sac; carries arm / scales / FACT path
    source_repo_id: str = ""
    source_root: str | None = None
    out_repo_id: str = ""
    out_root: str | None = None
    episodes: list[int] | None = None
    # Keep only frames whose phase (last entry of observation.state) is in this list; None = all frames.
    # The last kept frame of every episode is the terminal one.
    phases: list[int] | None = None
    # "auto": use `next.reward` if the source dataset has it, else reward 1 on the last frame of every
    # episode (all teleop demos are assumed successful).
    reward_source: str = "auto"
    batch_size: int = 32
    num_workers: int = 4
    fps: int = 10
    # See the module docstring. None (either, or both) = use observation.state / action unchanged.
    state_keep_names: list[str] | None = None
    action_keep_names: list[str] | None = None


def _percentiles(x: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(x.float(), torch.tensor([0.5, 0.95, 0.99, 1.0]))
    return {"p50": q[0].item(), "p95": q[1].item(), "p99": q[2].item(), "max": q[3].item()}


@parser.wrap()
def main(cfg: ConvertDemosConfig):
    init_logging()
    pcfg = cfg.policy
    if pcfg is None or pcfg.type != "residual_sac" or not pcfg.fact_pretrained_path:
        raise ValueError("Pass --policy.type=residual_sac and --policy.fact_pretrained_path=...")
    device = pcfg.device

    fact = FrozenFACT.from_pretrained(
        pcfg.fact_pretrained_path,
        device=device,
        feature_layer=pcfg.fact_feature_layer,
        pool_mode=pcfg.fact_pool_mode,
    )
    if fact.feature_dim != pcfg.fact_feature_dim:
        raise ValueError(
            f"policy.fact_feature_dim={pcfg.fact_feature_dim} doesn't match this checkpoint's actual "
            f"pooled feature size for pool_mode={pcfg.fact_pool_mode!r}: {fact.feature_dim}. "
            f"Pass --policy.fact_feature_dim={fact.feature_dim}."
        )
    space = ResidualActionSpace.from_config(pcfg)

    ds = LeRobotDataset(cfg.source_repo_id, root=cfg.source_root, episodes=cfg.episodes)
    if cfg.state_keep_names is not None or cfg.action_keep_names is not None:
        ds = SubsetStateActionDataset(
            ds,
            state_keep_names=cfg.state_keep_names or ds.meta.features["observation.state"]["names"],
            action_keep_names=cfg.action_keep_names or ds.meta.features["action"]["names"],
        )
        logging.info(
            "Subsetting observation.state/action to match the frozen FACT checkpoint's training schema: "
            "state %d -> %d dims, action %d -> %d dims",
            len(ds.base.meta.features["observation.state"]["names"]),
            len(ds.meta.features["observation.state"]["names"]),
            len(ds.base.meta.features["action"]["names"]),
            len(ds.meta.features["action"]["names"]),
        )
    has_reward = REWARD in ds.features
    use_dataset_reward = cfg.reward_source == "dataset" or (cfg.reward_source == "auto" and has_reward)
    logging.info("reward: %s", "dataset next.reward" if use_dataset_reward else "1 on last frame of episode")

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    keep_all, feats, res, ep_idx, frame_idx, rewards = [], [], [], [], [], []
    raw_pos, raw_grip, raw_rot = [], [], []
    for batch in tqdm(loader, desc="FACT features + residual labels"):
        if cfg.phases is not None:
            keep = torch.isin(batch["observation.state"][:, -1].round().long(), torch.tensor(cfg.phases))
            if not keep.any():
                continue
        else:
            keep = torch.ones(len(batch[ACTION]), dtype=torch.bool)

        obs = {k: v for k, v in batch.items() if k.startswith("observation.") or k == ACTION}
        out = fact(obs)  # FACT preprocessor moves data to the device
        a_demo = dataset_action_to_arms(batch[ACTION].to(out.feature.device).float())
        label = space.encode(out.arms, a_demo)

        # unclipped statistics (physical units) to help choosing residual_*_scale / max_residual_rot_rad.
        # raw_grip is still reported even when `include_gripper=False` -- it's a good sanity check that the
        # gripper really is close to constant in the demos (i.e. that excluding it was the right call).
        il, dm = out.arms[space.arm], a_demo[space.arm]
        raw_pos.append((dm.pos - il.pos).norm(dim=-1).cpu()[keep])
        raw_grip.append((dm.grip - il.grip).abs().squeeze(-1).cpu()[keep])
        r_res = inverse_compose_rotation(il.rot, dm.rot, space.rotation_compose)
        raw_rot.append(matrix_to_axis_angle(r_res).norm(dim=-1).cpu()[keep])

        feats.append(out.feature.cpu()[keep])
        res.append(label.cpu()[keep])
        ep_idx.append(batch["episode_index"][keep])
        frame_idx.append(batch["frame_index"][keep])
        if use_dataset_reward:
            rewards.append(batch[REWARD].reshape(len(keep), -1)[:, 0][keep].float())

    feats, res = torch.cat(feats), torch.cat(res)
    ep_idx, frame_idx = torch.cat(ep_idx), torch.cat(frame_idx)
    rewards = torch.cat(rewards) if use_dataset_reward else None
    n = len(feats)

    stats = {
        "num_frames": n,
        "arm": space.arm,
        "rotation_compose": space.rotation_compose,
        "include_gripper": space.include_gripper,
        "scales": {"pos": space.pos_scale, "grip": space.grip_scale, "max_rot_rad": space.max_rot_rad},
        "demo_residual_pos_norm": _percentiles(torch.cat(raw_pos)),
        "demo_residual_grip_abs": _percentiles(torch.cat(raw_grip)),
        "demo_residual_rot_angle_rad": _percentiles(torch.cat(raw_rot)),
        "label_clipped_fraction": {
            "pos": (res[:, :3].abs() >= 1.0).any(-1).float().mean().item(),
        },
    }
    if space.include_gripper:
        stats["label_clipped_fraction"]["grip"] = (res[:, 3].abs() >= 1.0).float().mean().item()
    logging.info("residual statistics:\n%s", json.dumps(stats, indent=2))
    if any(frac > 0.05 for frac in stats["label_clipped_fraction"].values()):
        logging.warning(
            "More than 5%% of the demo residual labels are clipped to [-1, 1]: increase "
            "policy.residual_pos_scale / residual_grip_scale (see p95/p99 above)."
        )

    buffer = ReplayBuffer(
        capacity=n, device="cpu", state_keys=[OBS_FACT_FEATURE], storage_device="cpu", use_drq=False
    )
    order = torch.argsort(ep_idx * 10_000_000 + frame_idx)  # episode-contiguous, time-ordered
    ep_sorted = ep_idx[order]
    for k, i in enumerate(order.tolist()):
        last = k == n - 1 or ep_sorted[k + 1] != ep_sorted[k]
        nxt = order[k + 1].item() if not last else i
        reward = float(rewards[i]) if use_dataset_reward else (1.0 if last else 0.0)
        buffer.add(
            state={OBS_FACT_FEATURE: feats[i][None]},
            action=res[i][None],
            reward=reward,
            next_state={OBS_FACT_FEATURE: feats[nxt][None]},
            done=bool(last),
            truncated=False,
        )

    out_ds = buffer.to_lerobot_dataset(
        repo_id=cfg.out_repo_id, fps=cfg.fps, root=cfg.out_root, task_name="peg_in_hole_residual"
    )
    (Path(out_ds.root) / "residual_stats.json").write_text(json.dumps(stats, indent=2))
    logging.info("Wrote %d residual transitions (%d episodes) to %s", n, len(ep_sorted.unique()), out_ds.root)


if __name__ == "__main__":
    main()
