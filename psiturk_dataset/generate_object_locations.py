#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import sys
import gzip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
import json

import habitat
from habitat.sims import make_sim


ISLAND_RADIUS_LIMIT = 1.5
VISITED_POINT_DICT = {}


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.
    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s, t, sim, near_dist, far_dist, geodesic_to_euclid_ratio
):
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0
    d_separation = sim.geodesic_distance(s, [t])
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0
    return True, d_separation


def get_random_object_receptacle_pair(object_to_receptacle_map):
    object_to_receptacle_map_length = len(object_to_receptacle_map)
    index = np.random.choice(object_to_receptacle_map_length)
    print(object_to_receptacle_map[index])
    return object_to_receptacle_map[index]


def get_object_receptacle_list(object_to_receptacle_map):
    object_to_receptacle_list = []
    for object_, receptacles in object_to_receptacle_map.items():
        for receptacle in receptacles:
            object_to_receptacle_list.append((object_, receptacle))
    return object_to_receptacle_list


def get_object_receptacle_pair(object_to_receptacle_list, index):
    index = index % len(object_to_receptacle_list)
    return object_to_receptacle_list[index]


def get_task_config(config, object_name, receptacle_name, object_ids, receptacle_ids):
    task = {}
    task["instruction"] = config["TASK"]["INSTRUCTION"].format(object_name, receptacle_name)
    task["type"] = config["TASK"]["TYPE"]
    task["goals"] = {}

    object_to_receptacle_map = {}
    for object_id, receptacle_id in zip(object_ids, receptacle_ids):
        if object_to_receptacle_map.get(object_id):
            object_to_receptacle_map[object_id].append(receptacle_id)
        else:
            object_to_receptacle_map[object_id] = [receptacle_id]

    task["goals"]["objectToRecepacleMap"] = object_to_receptacle_map
    return task


def build_episode(config, episode_id, objects, agent_position, agent_rotation, object_name, receptacle_name):
    scene_id = config.SIMULATOR.SCENE.split("/")[-1]
    task_config = config.TASK
    episode = {}
    episode["episode_id"] = episode_id
    episode["scene_id"] = scene_id
    episode["start_position"] = agent_position
    episode["start_rotation"] = agent_rotation

    object_ids = []
    receptacle_ids = []
    for object_ in objects:
        if object_["isReceptacle"]:
            object_ids.append(object_["objectId"])
        else:
            receptacle_ids.append(object_["objectId"])
    
    episode["task"] = get_task_config(config, object_name, receptacle_name, object_ids, receptacle_ids)
    episode["objects"] = objects
    return episode



def build_object(object_handle, object_id, object_name, is_receptacle, position, rotation):
    object_ = {
        "object": object_name,
        "objectHandle": "/data/objects/{}.phys_properties.json".format(object_handle),
        "objectIcon": "/data/test_assets/objects/{}.png".format(object_handle),
        "objectId": object_id,
        "isReceptacle": is_receptacle,
        "position": position,
        "rotation": rotation,
        "motionType": "DYNAMIC"
    }
    return object_


def get_bad_points(sim, points, d_lower_lim, d_upper_lim, geodesic_to_euclid_min_ratio, xlim=None, ylim=None, zlim=None):
    bad_points = np.zeros(points.shape[0], dtype=bool)
    # Outside X, Y, or Z limits
    if xlim:
        bad_points[points[:, 0] < xlim[0]] = 1
        bad_points[points[:, 0] > xlim[1]] = 1

    if ylim:
        bad_points[points[:, 1] < ylim[0]] = 1
        bad_points[points[:, 1] > ylim[1]] = 1

    if zlim:
        bad_points[points[:, 2] < zlim[0]] = 1
        bad_points[points[:, 2] > zlim[1]] = 1
    
    for i, point in enumerate(points):
        if VISITED_POINT_DICT.get(str(point)) == 1 or is_less_than_island_radius_limit(sim, point):
            bad_points[i] = 1

    # Too close to another object or receptacle
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i == j:
                continue

            is_compatible, dist = is_compatible_episode(
                point1,
                point2,
                sim,
                near_dist=d_lower_lim,
                far_dist=d_upper_lim,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )

            if not is_compatible:
                bad_points[i] = 1

    return bad_points


def rejection_sampling(
    sim, points, d_lower_lim, d_upper_lim,
    geodesic_to_euclid_min_ratio, xlim=None,
    ylim=None, zlim=None, num_tries=10000
):
    bad_points = get_bad_points(
        sim, points, d_lower_lim, d_upper_lim,
        geodesic_to_euclid_min_ratio, xlim, ylim, zlim
    )

    while sum(bad_points) > 0 and num_tries > 0:
        # print(sum(bad_points), num_tries)

        for i, bad_point in enumerate(bad_points):
            if bad_point:
                points[i] = sim.sample_navigable_point()

        bad_points = get_bad_points(
            sim, points, d_lower_lim, d_upper_lim,
            geodesic_to_euclid_min_ratio, xlim, ylim, zlim
        )
        num_tries -= 1
    
    print(sum(bad_points), num_tries)

    return points


def get_random_point(sim):
    point = sim.sample_navigable_point()
    return point


def get_random_rotation():
    angle = np.random.uniform(0, 2 * np.pi)
    rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    return rotation


def is_less_than_island_radius_limit(sim, point):
    return sim.island_radius(point) < ISLAND_RADIUS_LIMIT
        

def generate_points(
    config,
    objs_per_rec,
    num_episodes,
    num_targets,
    number_retries_per_target,
    d_lower_lim=0.5,
    d_upper_lim=50.0,
    geodesic_to_euclid_min_ratio=1.1
):
    # Initialize simulator
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    episode_count = 0
    episodes = []
    
    object_to_receptacle_list = get_object_receptacle_list(config["TASK"]["OBJECTS_RECEPTACLE_MAP"])
    object_name_map = dict(config["TASK"]["OBJECT_NAME_MAP"])
    y_limit = config["TASK"]["Y_LIMIT"]
    num_points = config["TASK"]["NUM_OBJECTS"] + config["TASK"]["NUM_RECEPTACLES"] + 1
    object_receptacle_pair_index = 0

    all_points = []

    while episode_count < num_episodes or num_episodes < 0:
        print("\nEpisode {}\n".format(episode_count))
        objects = []
        object_, receptacle = get_object_receptacle_pair(
            object_to_receptacle_list,
            object_receptacle_pair_index
        )

        object_name = object_name_map[object_]
        receptacle_name = object_name_map[receptacle]

        points = []
        rotations = []
        for idx in range(num_points):
            point = get_random_point(sim)
            rotation = get_random_rotation()
            points.append(point)
            rotations.append(rotation)
        
        points = rejection_sampling(
            sim, np.array(points), d_lower_lim, d_upper_lim,
            geodesic_to_euclid_min_ratio, ylim=y_limit
        )

        # Mark valid points as visited to get unique points
        for point in points:
            VISITED_POINT_DICT[str(point)] = 1
            all_points.append(point.tolist())
            
        agent_position = points[0].tolist()
        agent_rotation = rotations[0]
 
        target_position = points[1].tolist()
        target_rotation = rotations[1]

        source_position = points[2].tolist()
        source_rotation = rotations[2]

        # Create episode object configs
        objects.append(build_object(object_, len(objects), object_name, False, source_position, source_rotation))
        objects.append(build_object(receptacle, len(objects), receptacle_name, True, target_position, target_rotation))
        
        # Build episode from object and agent initilization.
        episode = build_episode(config, episode_count, objects, agent_position,
            agent_rotation, object_name, receptacle_name)
        episodes.append(episode)
        print(episode)

        object_receptacle_pair_index += 1
        episode_count += 1
    
    with open("points.csv", "w") as f:
        f.write(json.dumps(all_points))
    dataset = {
        "episodes": episodes
    }
    return dataset


def write_episode(dataset, output_path):
    with open(output_path, "w") as output_file:
        output_file.write(json.dumps(dataset))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate a new messy scene."
    )
    parser.add_argument(
        "--task-config",
        default="configs/generate_messyroom.yaml",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument(
        "--scenes",
        help="Scenes"
    )
    parser.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        default=2,
        help="Number of episodes to generate",
    )
    parser.add_argument(
        "-g",
        "--num_targets",
        type=int,
        default=10,
        help="Number of target points to sample",
    )
    parser.add_argument(
        "--number_retries_per_target",
        type=int,
        default=10,
        help="Number of retries for each target",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="episode_1.json",
        help="Output file for episodes",
    )
    parser.add_argument(
        "--d_lower_lim",
        type=float,
        default=0.5,
        help="Closest distance between objects allowed.",
    )
    parser.add_argument(
        "--d_upper_lim",
        type=float,
        default=30.0,
        help="Farthest distance between objects allowed.",
    )
    parser.add_argument(
        "--geodesic_to_euclid_min_ratio",
        type=float,
        default=1.1,
        help="Geodesic shortest path to Euclid distance ratio upper limit till aggressive sampling is applied.",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=1,
        help="Number of objects per goal.",
    )

    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)

    dataset_type = config.DATASET.TYPE
    if args.scenes is not None:
        config.defrost()
        config.SIMULATOR.SCENE = args.scenes
        config.freeze()

    if dataset_type == "Interactive":
        dataset = generate_points(
            config,
            args.ratio,
            args.num_episodes,
            args.num_targets,
            args.number_retries_per_target,
            args.d_lower_lim,
            args.d_upper_lim,
            args.geodesic_to_euclid_min_ratio
        )
        write_episode(dataset, args.output)
    else:
        print(f"Unknown dataset type: {dataset_type}")