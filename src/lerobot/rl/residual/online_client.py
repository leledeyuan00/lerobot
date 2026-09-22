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
"""The non-rollout half of `lerobot.rl.actor`, for an external control loop (e.g. a ROS2 node) that drives
the robot itself and only needs to (a) get residual actions from `ResidualPolicyRunner` and (b) report
transitions back to the (unmodified) `lerobot.rl.learner` for online SAC.

    python -m lerobot.rl.learner --config_path <train_config.json>   # start first, as usual

    client = ResidualOnlineClient(cfg, runner)   # runner: a ResidualPolicyRunner
    client.start()
    for episode in ...:
        runner.reset()
        while ...:
            obs = ...                                      # from ROS2 topics
            out = runner.select_action(obs)
            publish(out["action"])                          # to the impedance controller
            ...
            client.record_transition(
                state_feature=out["feature"], action=out["residual"], reward=..., next_state_feature=...,
                done=..., truncated=...,
            )
        client.end_episode(stats={"Episodic reward": ...})   # flushes the episode + pulls latest actor weights
    client.stop()

Runs the gRPC I/O (`receive_policy` / `send_transitions` / `send_interactions`, unmodified from
`lerobot.rl.actor`) in background threads, so the control loop calling `record_transition` never blocks on
the network -- matching CLAUDE.md's "actor/learner process split... for real-robot online RL without
blocking the control loop", just without the in-process env/rollout half of it (that part is your ROS2 node).
"""

from __future__ import annotations

import logging
import threading

import torch
from torch.multiprocessing import Queue

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.residual_sac.configuration_residual_sac import OBS_FACT_FEATURE
from lerobot.policies.residual_sac.runner import ResidualPolicyRunner
from lerobot.rl.actor import (
    establish_learner_connection,
    learner_service_client,
    push_transitions_to_transport_queue,
    receive_policy,
    send_interactions,
    send_transitions,
    update_policy_parameters,
)
from lerobot.transport.utils import python_object_to_bytes
from lerobot.utils.transition import Transition


class ResidualOnlineClient:
    def __init__(self, cfg: TrainRLServerPipelineConfig, runner: ResidualPolicyRunner):
        if cfg.policy.type != "residual_sac":
            raise ValueError(f"ResidualOnlineClient needs policy.type=residual_sac, got {cfg.policy.type}")
        self.cfg = cfg
        self.runner = runner
        self.device = runner.fact.device

        self._parameters_queue: Queue = Queue()
        self._transitions_queue: Queue = Queue()
        self._interactions_queue: Queue = Queue()
        self._episode: list[Transition] = []
        self._threads: list[threading.Thread] = []
        self._shutdown = threading.Event()
        # Running count of `record_transition()` calls, mirroring `lerobot.rl.actor`'s own global
        # `interaction_step` counter -- see the note on `end_episode()` for why this has to be sent.
        self._interaction_step = 0

    def start(self) -> None:
        """Connects to the learner and starts the background gRPC send/receive threads."""
        learner_client, grpc_channel = learner_service_client(
            host=self.cfg.policy.actor_learner_config.learner_host,
            port=self.cfg.policy.actor_learner_config.learner_port,
        )
        if not establish_learner_connection(learner_client, self._shutdown):
            raise RuntimeError("Could not establish a connection with the learner.")

        for fn, queue in (
            (receive_policy, self._parameters_queue),
            (send_transitions, self._transitions_queue),
            (send_interactions, self._interactions_queue),
        ):
            t = threading.Thread(target=fn, args=(self.cfg, queue, self._shutdown, grpc_channel), daemon=True)
            t.start()
            self._threads.append(t)
        logging.info("[ResidualOnlineClient] connected to learner, background threads started")

    def stop(self) -> None:
        self._shutdown.set()
        for q in (self._transitions_queue, self._interactions_queue, self._parameters_queue):
            q.close()
        for t in self._threads:
            t.join(timeout=5)

    def record_transition(
        self,
        state_feature: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state_feature: torch.Tensor,
        done: bool,
        truncated: bool,
        is_intervention: bool = False,
    ) -> None:
        """Buffers one `(s, a_residual, r, s', done)` step of the current episode.

        `state_feature` / `next_state_feature`: `ResidualPolicyRunner.select_action(...)["feature"]` (pooled
        FACT feature). `action`: the residual actually executed -- `out["residual"]` normally, or
        `runner.encode_intervention(human_action)` on a step where a human took over (do not store the raw
        16-d human action itself: the replay buffer / critic only ever see residuals).

        `is_intervention` is stored as a plain float under the string key `"is_intervention"`, for your own
        logging/filtering -- NOT under the `TeleopEvents.IS_INTERVENTION` enum key that (unmodified)
        `lerobot.rl.learner.process_transitions` checks to route a transition into the offline replay buffer.
        That enum-keyed check turns out to be unreachable for any transition that actually goes over the
        wire: transitions are (de)serialized with `torch.save`/`torch.load(weights_only=True)`
        (`lerobot.transport.utils.bytes_to_transitions`), which rejects an `Enum` member as a *dict key*
        unless it's been explicitly allow-listed (`torch.serialization.add_safe_globals`) -- on both the
        sending and the receiving process -- and nothing in this repo does that. `lerobot.rl.actor.py` never
        actually populates that key for the same reason, so this isn't a regression, just a pre-existing gap
        in the code this task reuses; it isn't papered over here rather than silently worked around.
        """
        self._episode.append(
            Transition(
                state={OBS_FACT_FEATURE: state_feature[None].cpu()},
                action=action[None].cpu(),
                reward=float(reward),
                next_state={OBS_FACT_FEATURE: next_state_feature[None].cpu()},
                done=bool(done),
                truncated=bool(truncated),
                complementary_info={"is_intervention": float(is_intervention)},
            )
        )
        self._interaction_step += 1

    def end_episode(self, stats: dict[str, float] | None = None) -> None:
        """Flushes the buffered episode to the learner, reports `stats`, and pulls the latest actor weights
        into `self.runner.policy` in place (whatever the learner has pushed since the last call).

        Always sends an interactions message, even if `stats` is empty -- **found by an integration dry run,
        not from the unit tests (which mock the learner)**: (unmodified) `lerobot.rl.learner`'s
        `process_interaction_message` does `message["Interaction step"] += interaction_step_shift`
        unconditionally, so a stats dict missing that key crashes the learner's training thread outright
        (`KeyError: 'Interaction step'`) the first time any episode's stats reach it -- not a slow-path
        gRPC issue, it kills the online loop on episode 1. `lerobot.rl.actor`'s own rollout loop always
        includes it (a running count of control-loop steps); `record_transition()` increments
        `self._interaction_step` to mirror that, so callers don't need to know this HIL-SERL-internal
        contract -- passing `stats=None` / `{}` still works, `stats` just adds on top (e.g. `Episodic reward`).
        """
        if self._episode:
            push_transitions_to_transport_queue(self._episode, self._transitions_queue)
            self._episode = []
        self._interactions_queue.put(
            python_object_to_bytes({"Interaction step": self._interaction_step, **(stats or {})})
        )
        update_policy_parameters(policy=self.runner.policy, parameters_queue=self._parameters_queue, device=self.device)
