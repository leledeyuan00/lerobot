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

from concurrent import futures
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.policies.residual_sac.configuration_residual_sac import OBS_FACT_FEATURE, ResidualSACConfig
from lerobot.policies.residual_sac.modeling_residual_sac import ResidualSACPolicy
from lerobot.rl.residual.online_client import ResidualOnlineClient
from lerobot.transport.utils import bytes_to_python_object, bytes_to_transitions
from tests.utils import require_package

D = 32


def _cfg(port: int = 50051):
    from lerobot.configs.train import TrainRLServerPipelineConfig

    policy = ResidualSACConfig(device="cpu", fact_feature_dim=D, push_to_hub=False)
    policy.actor_learner_config.learner_host = "localhost"
    policy.actor_learner_config.learner_port = port
    return TrainRLServerPipelineConfig(
        policy=policy,
        env=HILSerlRobotEnvConfig(),
        dataset=DatasetConfig(repo_id="unused"),
        output_dir="/tmp/residual_online_client_test",
    )


def _fake_runner():
    runner = MagicMock()
    runner.fact.device = torch.device("cpu")
    runner.policy = ResidualSACPolicy(ResidualSACConfig(device="cpu", fact_feature_dim=D, push_to_hub=False))
    return runner


def test_record_transition_buffers_and_end_episode_flushes_without_a_server():
    client = ResidualOnlineClient(_cfg(), _fake_runner())
    for i in range(3):
        client.record_transition(
            state_feature=torch.randn(D),
            action=torch.rand(10) * 2 - 1,
            reward=float(i == 2),
            next_state_feature=torch.randn(D),
            done=(i == 2),
            truncated=False,
            is_intervention=(i == 1),
        )
    assert len(client._episode) == 3
    client.end_episode(stats={"Episodic reward": 1.0})  # must not raise even with an empty parameters queue

    assert client._episode == []
    serialized = client._transitions_queue.get(timeout=2)
    transitions = bytes_to_transitions(serialized)
    assert len(transitions) == 3
    assert transitions[0]["reward"] == 0.0 and transitions[2]["reward"] == 1.0
    assert transitions[2]["done"] is True
    assert transitions[1]["complementary_info"]["is_intervention"].item() == 1.0
    assert transitions[0]["complementary_info"]["is_intervention"].item() == 0.0
    for t in transitions:
        assert t["state"][OBS_FACT_FEATURE].shape == (1, D)

    stats = bytes_to_python_object(client._interactions_queue.get(timeout=2))
    assert stats["Interaction step"] == 3  # one per record_transition() call
    assert stats["Episodic reward"] == 1.0


def test_end_episode_with_no_buffered_transitions_still_reports_interaction_step():
    """Found by an integration dry run against a real `lerobot.rl.learner`, not by these mocked-server
    tests: (unmodified) learner code does `message["Interaction step"] += ...` unconditionally on every
    interactions message, so skipping this key (the old behavior when `stats` was falsy) crashes the
    learner's training thread outright on the very first episode. See `end_episode()`'s docstring."""
    client = ResidualOnlineClient(_cfg(), _fake_runner())
    client.end_episode()  # nothing recorded, nothing to flush -- but still must report interaction step
    assert client._transitions_queue.empty()
    stats = bytes_to_python_object(client._interactions_queue.get(timeout=2))
    assert stats == {"Interaction step": 0}


def test_end_episode_sends_a_message_the_real_learner_can_process_without_crashing():
    """Regression test for the exact bug above: feed `end_episode()`'s wire message through the real
    (unmodified) `lerobot.rl.learner.process_interaction_message` -- this must not raise `KeyError`."""
    from lerobot.rl.learner import process_interaction_message

    client = ResidualOnlineClient(_cfg(), _fake_runner())
    client.record_transition(
        state_feature=torch.randn(D),
        action=torch.rand(9) * 2 - 1,
        reward=1.0,
        next_state_feature=torch.randn(D),
        done=True,
        truncated=False,
    )
    client.end_episode(stats={"Episodic reward": 1.0})
    raw = client._interactions_queue.get(timeout=2)
    message = process_interaction_message(raw, interaction_step_shift=0)
    assert message["Interaction step"] == 1
    assert message["Episodic reward"] == 1.0


def test_rejects_non_residual_sac_config():
    from lerobot.configs.train import TrainRLServerPipelineConfig
    from lerobot.policies.sac.configuration_sac import SACConfig

    cfg = TrainRLServerPipelineConfig(
        policy=SACConfig(device="cpu"), env=HILSerlRobotEnvConfig(), output_dir="/tmp/x"
    )
    with pytest.raises(ValueError):
        ResidualOnlineClient(cfg, _fake_runner())


@require_package("grpc")
def test_start_stop_lifecycle_against_a_real_server():
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc

    class MockLearnerService(services_pb2_grpc.LearnerServiceServicer):
        def Ready(self, request, context):  # noqa: N802
            return services_pb2.Empty()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_LearnerServiceServicer_to_server(MockLearnerService(), server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    try:
        client = ResidualOnlineClient(_cfg(port=port), _fake_runner())
        client.start()
        assert len(client._threads) == 3
        client.stop()
        assert all(not t.is_alive() for t in client._threads)
    finally:
        server.stop(None)
