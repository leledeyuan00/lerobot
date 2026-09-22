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
"""Warm-up step 2: offline pre-training of the twin critics on the converted demo transitions.

Only the critics (and their target copies) are trained. The actor is NOT behaviour-cloned: it stays at its
no-op initialisation (near-zero weights, identity-rotation bias) and learns online from the warmed-up critic.
Load the result in the learner / actor with `--policy.pretrained_path=<output_dir>/pretrained_model`.

    python -m lerobot.rl.residual.warmup_critic --policy.type=residual_sac \
        --dataset.repo_id=leledeyuan/mating_cable_residual --dataset.root=/data/mating_cable_residual \
        --output_dir=outputs/warmup_critic --steps=5000 --wandb.enable=true --wandb.project=residual-sac

`--wandb.enable=true` logs scalars only (loss, Q diagnostics) via the same `WandBLogger` `lerobot.rl.learner`
uses -- it never uploads the policy itself as a WandB artifact (`log_policy()`, the method that would, is
simply never called here); the actual checkpoint is only ever written to `output_dir` on local/NAS disk.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.residual_sac.configuration_residual_sac import OBS_FACT_FEATURE
from lerobot.policies.residual_sac.modeling_residual_sac import ResidualSACPolicy, canonicalize_action
from lerobot.policies.residual_sac.residual_action import IDENTITY_ROT6D, RES_ROT
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class WarmupCriticConfig:
    policy: PreTrainedConfig | None = None
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id=""))
    output_dir: str = "outputs/warmup_critic"
    job_name: str | None = None
    steps: int = 5000
    batch_size: int = 256
    log_freq: int = 100
    seed: int = 1000
    # Stop early / warn when Q leaves [0, q_max]: rewards are in {0,1}, so Q > 1 means overestimation.
    q_max: float = 1.5
    wandb: WandBConfig = field(default_factory=WandBConfig)
    # WandBLogger's constructor is written against TrainPipelineConfig; these two are read (env for the
    # group name, resume for run-id continuity) but always empty/False here -- there's no env, and this
    # script doesn't support resuming a partial warm-up.
    env: None = None
    resume: bool = False

    def to_dict(self) -> dict:
        return draccus.encode(self)


@torch.no_grad()
def q_diagnostics(policy: ResidualSACPolicy, batch, q_max: float) -> dict[str, float]:
    """Q on demo residuals vs. on random (OOD) residuals: the gap shows whether the critic discriminates.

    Q includes the soft (entropy) backup term, so with a low-entropy actor it sits below the pure
    discounted-reward value (which is in [0, 1]); only the upper bound `q_max` is a divergence signal.
    """
    obs = batch["state"]
    b = batch["action"].shape[0]
    rand = torch.rand_like(batch["action"]) * 2 - 1
    rot = torch.randn(b, 6, device=rand.device) + torch.tensor(IDENTITY_ROT6D, device=rand.device)
    rand[:, RES_ROT] = rot
    rand = canonicalize_action(rand)
    q_demo = policy.critic_forward(obs, batch["action"]).min(dim=0).values
    q_rand = policy.critic_forward(obs, rand).min(dim=0).values
    return {
        "q_demo_mean": q_demo.mean().item(),
        "q_rand_mean": q_rand.mean().item(),
        "q_gap_demo_minus_rand": (q_demo - q_rand).mean().item(),
        "q_max": max(q_demo.max().item(), q_rand.max().item()),
        "q_overshoot": float(max(q_demo.max().item(), q_rand.max().item()) > q_max),
    }


@parser.wrap()
def main(cfg: WarmupCriticConfig):
    init_logging()
    set_seed(cfg.seed)
    pcfg = cfg.policy
    if pcfg is None or pcfg.type != "residual_sac":
        raise ValueError("Pass --policy.type=residual_sac")
    device = torch.device(pcfg.device)

    wandb_logger = None
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        logging.info("wandb disabled; logging locally only.")

    policy = ResidualSACPolicy(pcfg).to(device)
    policy.train()

    ds = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
    buffer = ReplayBuffer.from_lerobot_dataset(
        ds, device=str(device), state_keys=[OBS_FACT_FEATURE], use_drq=False, optimize_memory=False
    )
    logging.info("critic warm-up on %d converted demo transitions", len(buffer))

    opt = torch.optim.Adam(policy.critic_ensemble.parameters(), lr=pcfg.critic_lr)
    for step in range(1, cfg.steps + 1):
        b = buffer.sample(cfg.batch_size)
        loss = policy.forward(
            {
                "action": b["action"],
                "reward": b["reward"],
                "state": b["state"],
                "next_state": b["next_state"],
                "done": b["done"],
            },
            model="critic",
        )["loss_critic"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.critic_ensemble.parameters(), pcfg.grad_clip_norm)
        opt.step()
        policy.update_target_networks()

        if step % cfg.log_freq == 0 or step == cfg.steps:
            diag = q_diagnostics(policy, b, cfg.q_max)
            logging.info("step %d loss %.5f %s", step, loss.item(), {k: round(v, 4) for k, v in diag.items()})
            if diag["q_overshoot"]:
                logging.warning("Q-values exceed %.2f (max reward is 1): possible OOD overestimation.", cfg.q_max)
            if wandb_logger is not None:
                wandb_logger.log_dict({"loss_critic": loss.item(), **diag}, step=step, mode="train")

    out = Path(cfg.output_dir) / "pretrained_model"
    policy.save_pretrained(out)
    logging.info("Saved warmed-up policy to %s (use --policy.pretrained_path=%s)", out, out)


if __name__ == "__main__":
    main()
