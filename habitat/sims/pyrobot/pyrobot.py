#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from gym import Space, spaces

from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    DepthSensor,
    RGBSensor,
    SensorSuite,
    Simulator,
)
import pyrobot
import numpy as np

# TODO(akadian): remove the below pyrobot hack
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    import cv2
sys.path.append(ros_path)


@registry.register_sensor
class PyRobotRGBSensor(RGBSensor):
    def __init__(self, config):
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    def get_observation(self, robot_obs):
        obs = robot_obs.get(self.uuid, None) 

        assert obs is not None, "Invalid observation for {} sensor".format(self.uuid)

        if obs.shape != self.observation_space.shape:
            obs = cv2.resize(obs, (self.observation_space.shape[1], self.observation_space.shape[0]))

        return obs


@registry.register_sensor
class PyRobotDepthSensor(DepthSensor):
    def __init__(self, config):
        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(self, robot_obs):
        obs = robot_obs.get(self.uuid, None)

        assert obs is not None, "Invalid observation for {} sensor".format(self.uuid)

        if obs.shape != self.observation_space.shape:
            obs = cv2.resize(obs, (self.observation_space.shape[1], self.observation_space.shape[0]))

        obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
        if self.config.NORMALIZE_DEPTH:
            # normalize depth observations to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / self.config.MAX_DEPTH

        obs = np.expand_dims(obs, axis=2)  # make depth observations a 3D array

        return obs


@registry.register_simulator(name="PyRobot-v0")
class PyRobot(Simulator):
    def __init__(self, config: Config) -> None:
        self._config = config
        
        robot_sensors = []
        for sensor_name in self._config.SENSORS:
            sensor_cfg = getattr(self._config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)     

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            robot_sensors.append(sensor_type(sensor_cfg))
        self._sensor_suite = SensorSuite(robot_sensors)

        config_pyrobot = {
            "base_controller": self._config.BASE_CONTROLLER
        }

        assert self._config.ROBOT in self._config.ROBOTS, "Invalid robot type {}".format(self._config.ROBOT)
        self._robot_config = getattr(
            self._config, 
            self._config.ROBOT.upper()
        )

        self._robot = pyrobot.Robot(self._config.ROBOT, base_config=config_pyrobot)

    def _degree_to_radian(self, degrees):
        return (degrees / 180) * np.pi

    def _robot_obs(self):
        return {
            "rgb": self._robot.camera.get_rgb(),
            "depth": self._robot.camera.get_depth(),
        }

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def base(self):
        return self._robot.base

    @property
    def camera(self):
        return self._robot.camera

    # TODO(akadian): add action space support.

    def reset(self):
        self._robot.camera.reset()

        observations = self._sensor_suite.get_observations(self._robot_obs())
        return observations

    def step(self, action, action_params):
        if action in self._robot_config.BASE_ACTIONS:
            getattr(self._robot.base, action)(**action_params)
        elif action in self._robot_config.CAMERA_ACTIONS:
            getattr(self._robot.camera, action)(**action_params)
        else:
            raise ValueError("Invalid action {}".format(action))

        observations = self._sensor_suite.get_observations(self._robot_obs())

        return observations

    def seed(self, seed: int) -> None:
        raise NotImplementedError("No support for seeding in reality")