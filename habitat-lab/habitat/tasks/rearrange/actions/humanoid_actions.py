#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import magnum as mn
import numpy as np
import quaternion
from gym import spaces
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat.tasks.rearrange.actions.actions import HumanoidJointAction
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_from_two_vectors,
    quaternion_rotate_vector,
)
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_magnum


class HandState(Enum):
    APPROACHING = 0
    RETRACTING = 1


@registry.register_task_action
class HumanoidPickAction(HumanoidJointAction):
    def __init__(self, *args, task, **kwargs):
        config = kwargs["config"]
        HumanoidJointAction.__init__(self, *args, **kwargs)
        self.vdb = None

        self.humanoid_controller = self.lazy_inst_humanoid_controller(
            task, config
        )

        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self._prev_ep_id = None
        self._targets = {}
        self.skill_done = False
        self.hand_state = HandState.APPROACHING

        self.dist_move_per_step = config.dist_move_per_step
        self.dist_to_snap = config.dist_to_snap

        self.init_coord = mn.Vector3(
            0.2, 0.2, 0
        )  # Init coord with respect to the agent root pose.
        self.hand_pose_iter = 0

    def lazy_inst_humanoid_controller(self, task, config):
        # Lazy instantiation of humanoid controller
        # We assign the task with the humanoid controller, so that multiple actions can
        # use it.

        if (
            not hasattr(task, "humanoid_controller")
            or task.humanoid_controller is None
        ):
            # Initialize humanoid controller
            agent_name = self._sim.habitat_config.agents_order[
                self._agent_index
            ]
            walk_pose_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path

            humanoid_controller = HumanoidRearrangeController(walk_pose_path)
            humanoid_controller.set_framerate_for_linspeed(
                config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
            )
            task.humanoid_controller = humanoid_controller

        self.vdb = DebugVisualizer(self._sim, output_path="")
        return task.humanoid_controller

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "humanoid_pick_action": spaces.Box(
                    shape=(2,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _get_coord_for_idx(self, object_target_idx):
        pick_obj_entity = self._entities[object_target_idx]
        obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
            pick_obj_entity
        )
        return obj_pos

    def get_scene_index_obj(self, object_target_idx):
        pick_obj_entity = self._entities[object_target_idx]
        entity_name = pick_obj_entity.name
        obj_id = self._task.pddl_problem.sim_info.obj_ids[entity_name]
        return self._sim.scene_obj_ids[obj_id]

    def _direction_to_quaternion(
        self, direction_vector: np.ndarray, origin_vector: np.ndarray
    ):
        output = quaternion_from_two_vectors(origin_vector, direction_vector)
        output = output.normalized()
        return output

    def quaternion_to_euler(self, q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw) in radians.

        Parameters:
            q (quaternion): The quaternion to convert.

        Returns:
            tuple: Euler angles (roll, pitch, yaw) in radians.
        """
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion.as_rotation_matrix(q)

        # Extract Euler angles from rotation matrix
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(
            -rotation_matrix[2, 0],
            np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2),
        )
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
        return [roll, pitch * 0.75, 0]

    def calculate_turn_delta(self, sensor_object, cam_offset, goal_postion):
        agent_pose = sensor_object.node.translation  # + cam_offset
        agent_rotation = sensor_object.node.rotation

        target_pose = goal_postion
        if not isinstance(agent_rotation, quaternion.quaternion):
            rot = np.array(agent_rotation.vector).tolist() + [
                agent_rotation.scalar
            ]
            agent_rotation = quaternion_from_coeff(rot)

        position_delta = target_pose - agent_pose

        pos_without_y = np.array([position_delta[0], 0, position_delta[2]])
        rot_z = self._direction_to_quaternion(
            pos_without_y, np.array([0, 0, -1])
        )

        _, y, _ = self.quaternion_to_euler(rot_z)
        if abs(y) > 180:
            sign = 1 if y > 0 else -1
            new_y = 360 - abs(y)
            rot_z = quat_from_angle_axis(
                np.deg2rad(new_y) * sign * -1, np.array([0, 0, 1])
            )

        pos_without_x = np.array([0, position_delta[1], position_delta[2]])
        y_direction = abs(goal_postion[1]) - abs(agent_pose[1])
        y_direction = -1 if y_direction > 0 else 1
        rot_y = self._direction_to_quaternion(
            pos_without_x, np.array([0, y_direction, 0])
        )

        p, _, _ = self.quaternion_to_euler(rot_y)
        if abs(p) > 90:
            sign = 1 if p > 0 else -1
            rot_y = quat_from_angle_axis(1.57 * sign, np.array([1, 0, 0]))

        rotation = rot_z * rot_y
        rotation = self.quaternion_to_euler(rotation)
        return position_delta, rotation

    def patch_camera_orientation(self, target_pos):
        articulated_agent = self._sim.get_agent_data(0).articulated_agent
        for cam_prefix, sensor_names in articulated_agent._cameras.items():
            for sensor_name in sensor_names:
                sens_obj = self._sim._sensors[sensor_name]._sensor_object
                if "head_rgb" not in sensor_name:
                    continue

                cam_offset = articulated_agent._camera_metadata[sensor_name][
                    "look_at_offset"
                ]
                pos, rot_delta = self.calculate_turn_delta(
                    sens_obj,
                    cam_offset,
                    target_pos,
                )
                articulated_agent._camera_metadata[sensor_name] = {
                    "look_at_changed": True,
                    "look_at_offset": cam_offset,
                    "rot_delta": rot_delta,
                }

    def step(self, *args, **kwargs):
        self.skill_done = False
        object_pick_idx = (
            kwargs[self._action_arg_prefix + "humanoid_pick_action"][0] - 1
        )
        should_pick = kwargs[self._action_arg_prefix + "humanoid_pick_action"][
            1
        ]

        if object_pick_idx < 0 or object_pick_idx > len(self._entities):
            return

        object_coord = self._get_coord_for_idx(object_pick_idx)
        init_coord_world = (
            self.humanoid_controller.obj_transform_base.transform_point(
                self.init_coord
            )
        )

        hand_vector = (object_coord - init_coord_world) / np.linalg.norm(
            object_coord - init_coord_world
        )
        max_num_iters = int(
            np.linalg.norm(object_coord - init_coord_world)
            / self.dist_move_per_step
        )
        # if not should_pick:
        #     print("[Place] dist_hand_obj: ", self.hand_state, object_coord)

        should_rest = False
        if self.hand_state == HandState.APPROACHING:  # Approaching
            # Only move the hand to object if has to drop or object is not grabbed
            if should_pick == 0 or self.cur_grasp_mgr.snap_idx is None:
                new_hand_coord = (
                    init_coord_world
                    + self.hand_pose_iter
                    * self.dist_move_per_step
                    * hand_vector
                )
                self.hand_pose_iter = min(
                    self.hand_pose_iter + 1, max_num_iters
                )
                dist_hand_obj = np.linalg.norm(object_coord - new_hand_coord)
                # if not should_pick:
                #     print("[Place] dist_hand_obj: ", dist_hand_obj)
                if dist_hand_obj < self.dist_to_snap:
                    # snap,
                    self.hand_state = HandState.RETRACTING
                    if should_pick:
                        object_index = self.get_scene_index_obj(object_pick_idx)
                        if self.cur_grasp_mgr.snap_idx is None:
                            self.cur_grasp_mgr.snap_to_obj(
                                object_index,
                            )
                        self._sim.internal_step(-1)
                    else:
                        obj_grabbed = self.cur_grasp_mgr.snap_rigid_obj
                        self.cur_grasp_mgr.desnap(True)
                        if obj_grabbed is not None:
                            obj_grabbed.transformation = mn.Matrix4.translation(
                                object_coord
                            )
            else:
                should_rest = True

        else:  # Retracting
            new_hand_coord = (
                init_coord_world
                + self.hand_pose_iter * self.dist_move_per_step * hand_vector
            )
            self.hand_pose_iter = max(0, self.hand_pose_iter - 1)
            dist_hand_init = np.linalg.norm(new_hand_coord - init_coord_world)
            if dist_hand_init < self.dist_to_snap:
                self.hand_state = HandState.APPROACHING
                self.skill_done = True
                self.hand_pose_iter = 0

        if should_rest:
            self.humanoid_controller.calculate_stop_pose()
        else:
            self.humanoid_controller.calculate_reach_pose(new_hand_coord)

        base_action = self.humanoid_controller.get_pose()
        kwargs[f"{self._action_arg_prefix}human_joints_trans"] = base_action
        self.patch_camera_orientation(object_coord)

        HumanoidJointAction.step(self, *args, **kwargs)
        return

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id
            self.skill_done = False
        self.hand_pose_iter = 0
