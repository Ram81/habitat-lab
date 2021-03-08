#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type

import attr
import habitat_sim
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, Simulator, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import merge_sim_episode_config, DistanceToGoal
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import SimulatorTaskAction, Measure
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.utils import get_habitat_sim_action, get_habitat_sim_action_str
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: List[int]


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementSpec:
    r"""Specifications that capture a particular position of final position
    or initial position of the object.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[Dict[str, str]] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementObjectSpec(RearrangementSpec):
    r"""Object specifications that capture position of each object in the scene,
    the associated object template.
    """
    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_handle: Optional[str] = attr.ib(
        default="", validator=not_none_validator
    )
    object_template: Optional[str] = attr.ib(
        default="", validator=not_none_validator
    )
    object_icon: Optional[str] = attr.ib(default="")
    motion_type: Optional[str] = attr.ib(default=None)
    is_receptacle: Optional[bool] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class GrabReleaseActionSpec:
    r"""Grab/Release action reaply data specifications that capture states
     of each grab/release action.
    """
    new_object_translation: Optional[List[float]] = attr.ib(default=None)
    gripped_object_id: Optional[int] = attr.ib(default=None)
    new_object_id: Optional[int] = attr.ib(default=None)
    object_handle: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ObjectStateSpec:
    r"""Object data specifications that capture states of each object in replay state.
    """
    object_id: Optional[int] = attr.ib(default=None)
    translation: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    motion_type: Optional[str] = attr.ib(default=None)
    object_handle: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state.
    """
    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action.
    """
    action: str = attr.ib(default=None, validator=not_none_validator)
    object_under_cross_hair: Optional[int] = attr.ib(default=None)
    object_drop_point: Optional[List[float]] = attr.ib(default=None)
    action_data: Optional[GrabReleaseActionSpec] = attr.ib(default=None)
    is_grab_action: Optional[bool] = attr.ib(default=None)
    is_release_action: Optional[bool] = attr.ib(default=None)
    object_states: Optional[List[ObjectStateSpec]] = attr.ib(default=None)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)
    collision: Optional[dict] = attr.ib(default=None)
    timestamp: Optional[int] = attr.ib(default=None)
    nearest_object_id: Optional[int] = attr.ib(default=None)
    gripped_object_id: Optional[int] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementEpisode(Episode):
    r"""Specification of episode that includes initial position and rotation
    of agent, goal specifications, instruction specifications, reference path,
    and optional shortest paths.

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        instruction: single natural language instruction for the task.
        reference_replay: List of keypresses which gives the reference
            actions to the goal that aligns with the instruction.
    """
    goals: List[RearrangementSpec] = attr.ib(
        default=None, validator=not_none_validator
    )
    reference_replay: List[Dict] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    objects: List[RearrangementObjectSpec] = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=66,
            shape=(9,),
            dtype=np.int64,
        )

    def _get_sensor_type(self, *args:Any, **kwargs: Any):
        return SensorTypes.TOKEN_IDS

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: RearrangementEpisode,
        **kwargs
    ):
        return episode.instruction.instruction_tokens

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="DemonstrationSensor")
class DemonstrationSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "demonstration"
        self.observation_space = spaces.Discrete(0)
        self.timestep = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: RearrangementEpisode,
        task: EmbodiedTask,
        **kwargs
    ):
        if not task.is_episode_active:  # reset
            self.timestep = 0
        
        if self.timestep < len(episode.reference_replay):
            action_name = episode.reference_replay[self.timestep].action
            action = get_habitat_sim_action(action_name)
        else:
            action = -1

        # print("{} -- {}".format(self.timestep, get_habitat_sim_action_str(action)))
        self.timestep += 1
        return action

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="GrippedObjectSensor")
class GrippedObjectSensor(Sensor):
    def __init__(self, *args, sim: HabitatSim, config: Config, **kwargs):
        self._sim = sim
        self.uuid = "gripped_object_id"
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Discrete(
            len(self._sim.get_existing_object_ids())
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_sensor_type(self, *args:Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: RearrangementEpisode,
        *args: Any,
        **kwargs
    ):
        return self._sim.gripped_object_id

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_measure
class ObjectToReceptacleDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_receptacle_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToReceptacleDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()
        obj_id = -1
        receptacle_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object.is_receptacle == False:
                obj_id = scene_object.object_id
            else:
                receptacle_id = scene_object.object_id

        if receptacle_id == -1:
            self._metric = 100
        elif obj_id != -1:
            object_position = np.array(
                self._sim.get_translation(obj_id)
            ).tolist()

            receptacle_position = np.array(
                self._sim.get_translation(receptacle_id)
            ).tolist()

            self._metric = self._geo_dist(
                object_position, receptacle_position
            )
        else:
            receptacle_position = np.array(
                self._sim.get_translation(receptacle_id)
            ).tolist()

            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            self._metric = self._geo_dist(
                agent_position, receptacle_position
            )


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def _geo_dist(self, src_pos, object_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [object_pos])

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()

        sim_obj_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object.is_receptacle == False:
                sim_obj_id = scene_object.object_id

        if sim_obj_id != -1:
            previous_position = np.array(
                self._sim.get_translation(sim_obj_id)
            ).tolist()

            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            self._metric = self._geo_dist(
                previous_position, agent_position
            )
        else:
            self._metric = 0


@registry.register_measure
class AgentToReceptacleDistance(Measure):
    """The measure calculates the distance of receptacle from the agent"""

    cls_uuid: str = "agent_receptacle_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToReceptacleDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def _geo_dist(self, src_pos, object_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [object_pos])

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()

        sim_obj_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object.is_receptacle == True:
                sim_obj_id = scene_object.object_id

        recceptacle_position = np.array(
            self._sim.get_translation(sim_obj_id)
        ).tolist()

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position

        self._metric = self._geo_dist(
            recceptacle_position, agent_position
        )


@registry.register_measure
class RearrangementSuccess(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [ObjectToReceptacleDistance.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            ObjectToReceptacleDistance.cls_uuid
        ].get_metric()
        object_ids = self._sim.get_existing_object_ids()

        obj_id = -1
        receptacle_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object.is_receptacle == False:
                obj_id = scene_object.object_id
            else:
                receptacle_id = scene_object.object_id

        is_object_stacked = False
        if obj_id != -1 and receptacle_id != -1:
            object_position = self._sim.get_translation(obj_id)
            receptacle_position = self._sim.get_translation(receptacle_id)

            object_y = object_position.y
            receptacle_y = receptacle_position.y + self._sim.get_object_bb_y_coord(receptacle_id)
            is_object_stacked = (object_y > receptacle_y)

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called # type: ignore
            and distance_to_target <= self._config.SUCCESS_DISTANCE
            and is_object_stacked
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class RearrangementSPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, RearrangementSuccess.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[RearrangementSuccess.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class GrabSuccess(Measure):
    r"""Grab Success - whether an object was grabbed during episode or not
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.prev_gripped_object_id = -1

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "grab_success"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        gripped_object_id = self._sim.gripped_object_id
        if gripped_object_id != -1 and gripped_object_id != self.prev_gripped_object_id:
            self._metric = 1
        else:
            self._metric = 0
        self.prev_gripped_object_id = gripped_object_id


def merge_sim_episode_with_object_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()
    sim_config.objects = episode.objects
    sim_config.freeze()

    return sim_config


@registry.register_task(name="RearrangementTask-v0")
class RearrangementTask(EmbodiedTask):
    r"""Language based Object Rearrangement Task
    Goal: An agent must rearrange objects in a 3D environment
        specified by a natural language instruction.
    Usage example:
        examples/object_rearrangement_example.py
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False
    
    def reset(self, **kwargs):
        self._is_episode_active = False
        observations = super().reset(**kwargs)
        self._is_episode_active = True
        return observations

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)
