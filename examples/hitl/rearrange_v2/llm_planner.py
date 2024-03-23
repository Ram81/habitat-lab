#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.planner import CentralizedPlanner
from omegaconf import DictConfig


class PlannerWrapper:
    def __init__(
        self, config: DictConfig, env_interface: EnvironmentInterface
    ) -> None:
        self.config = config
        self.env_interface = env_interface

        self.planner = CentralizedPlanner(config.planner, env_interface)

    def run(self, observations, episode):
        episode_id = episode.episode_id

        r, step_count, llm_call_count, info = self.planner(
            output_name=f"episode_{episode_id}_0"
        )

        info_episode = {
            "run_id": 0,
            "episode_id": episode_id,
            "instruction": episode.info["extra_info"]["instruction"],
        }

        stats_keys = {
            "task_percent_complete",
            "task_state_success",
            "num_steps",
        }

        # Reset env_interface (moves onto the next episode in the dataset)
        self.env_interface.reset_environment()

        # Reset planner
        self.planner.reset()
