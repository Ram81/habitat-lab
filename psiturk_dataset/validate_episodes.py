#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import gzip
import habitat
import json
import os
import random
import scipy
import sys

import numpy as np
import magnum as mn
import matplotlib.pyplot as plt


from habitat.sims import make_sim
from habitat_sim.utils.common import quat_from_coeffs, quat_from_magnum, quat_to_coeffs
from mpl_toolkits.mplot3d import Axes3D


ISLAND_RADIUS_LIMIT = 1.5
VISITED_POINT_DICT = {}


def get_geodesic_distance(sim, position_a, position_b):
    return sim.geodesic_distance(position_a, position_b)


def populate_episodes_points(episodes, scene_id):
    for episode in episodes:
        if scene_id != episode["scene_id"]:
            continue
        point = str(episode["start_position"])
        if VISITED_POINT_DICT.get(point):
            VISITED_POINT_DICT[point] += 1
            print("Redundant agent position in episode {}".format(episode["episode_id"]))
        else:
            VISITED_POINT_DICT[point] = 1

        for object_ in episode["objects"]:
            point = str(object_["position"])
            if VISITED_POINT_DICT.get(point):
                VISITED_POINT_DICT[point] += 1
                print("Redundant point in episode {}".format(episode["episode_id"]))
            else:
                VISITED_POINT_DICT[point] = 1


def is_valid_episode(sim, episode, near_dist, far_dist):
    agent_position = episode["start_position"]
    positions = [agent_position]
    for object_ in episode["objects"]:
        position = object_["position"]
        positions.append(position)

    for i in range(len(positions)):
        for j in range(len(positions)):
            if i <= j:
                continue
            dist = get_geodesic_distance(sim, positions[i], positions[j])
            if not near_dist <= dist <= far_dist:
                return False
    return True
    

def get_all_tasks(path, scene_id):
    tasks = []
    for file_path in glob.glob(path + "/*.json"):
        with open(file_path, "r") as file:
            data = json.loads(file.read())
            if data["episodes"][0]["scene_id"] == scene_id:
                tasks.append(data)
                populate_episodes_points(data["episodes"], scene_id)
                print(file_path)
    unique_points_count = len(VISITED_POINT_DICT.keys())
    print("Total tasks: {}".format(len(tasks)))
    print("Total unique points: {} -- {}".format(unique_points_count, unique_points_count / 3))
    return tasks


def get_sim(config):
    # Initialize simulator
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    return sim


def validate_tasks(
    config,
    d_lower_lim=5.0,
    d_upper_lim=30.0,
    prev_episodes="data/tasks",
    scene_id="empty_house.glb"
):
    sim = get_sim(config)

    # Populate previously generated points
    tasks = get_all_tasks(prev_episodes, scene_id)

    results = []
    i = 0
    for task in tasks:
        episodes = task["episodes"]
        count = 0
        for episode in episodes:
            is_valid = is_valid_episode(sim, episode, d_lower_lim, d_upper_lim)
            count += int(is_valid)
        i += 1

        print("\nScene: {}, Num valid episodes: {}, Total episodes: {}\n".format(scene_id, count, len(episodes)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate new episodes."
    )
    parser.add_argument(
        "--task-config",
        default="psiturk_dataset/rearrangement.yaml",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument(
        "--scenes",
        help="Scenes",
        default="data/scene_datasets/habitat-test-scenes/empty_house.glb"
    )
    parser.add_argument(
        "--d_lower_lim",
        type=float,
        default=5,
        help="Closest distance between objects allowed.",
    )
    parser.add_argument(
        "--d_upper_lim",
        type=float,
        default=30.0,
        help="Farthest distance between objects allowed.",
    )
    parser.add_argument(
        "--prev_episodes",
        default="data/tasks",
        help="Task configuration file for initializing a Habitat environment",
    )

    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)

    dataset_type = config.DATASET.TYPE
    scene_id = ""
    if args.scenes is not None:
        config.defrost()
        config.SIMULATOR.SCENE = args.scenes
        config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
        config.freeze()
        scene_id = args.scenes.split("/")[-1]

    if dataset_type == "Interactive":
        validate_tasks(
            config,
            args.d_lower_lim,
            args.d_upper_lim,
            args.prev_episodes,
            scene_id
        )
    else:
        print(f"Unknown dataset type: {dataset_type}")