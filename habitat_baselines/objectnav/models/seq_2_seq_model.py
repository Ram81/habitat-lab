import math
import sys
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoalSensor
)
from habitat_baselines.rearrangement.models.encoders.instruction import InstructionEncoder
from habitat_baselines.rearrangement.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
)
from habitat_baselines.rearrangement.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet
from habitat_baselines.utils.common import CategoricalNet, CustomFixedCategorical


class Seq2SeqNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions, obs_augmentations):
        super().__init__()
        self.model_config = model_config
        rnn_input_size = 0

        if not self.model_config.NO_VISION:
            # Init the depth encoder
            assert model_config.DEPTH_ENCODER.cnn_type in [
                "SimpleDepthCNN",
                "VlnResnetDepthEncoder",
                "None"
            ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
            logger.info("depth encoder: {}".format(model_config.DEPTH_ENCODER.cnn_type))
            if model_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
                self.depth_encoder = SimpleDepthCNN(
                    observation_space, model_config.DEPTH_ENCODER.output_size
                )
                rnn_input_size += model_config.DEPTH_ENCODER.output_size
            elif model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
                self.depth_encoder = VlnResnetDepthEncoder(
                    observation_space,
                    output_size=model_config.DEPTH_ENCODER.output_size,
                    checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                    backbone=model_config.DEPTH_ENCODER.backbone,
                    trainable=model_config.DEPTH_ENCODER.trainable,
                )
                rnn_input_size += model_config.DEPTH_ENCODER.output_size
            else:
                self.depth_encoder = None
                logger.info("depth encoder is None: {}".format(self.depth_encoder))

            # Init the RGB visual encoder
            assert model_config.RGB_ENCODER.cnn_type in [
                "SimpleRGBCNN",
                "TorchVisionResNet50",
                "ResnetRGBEncoder",
                "None"
            ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

            if model_config.RGB_ENCODER.cnn_type == "SimpleRGBCNN":
                self.rgb_encoder = SimpleRGBCNN(
                    observation_space, model_config.RGB_ENCODER.output_size
                )
                rnn_input_size += model_config.RGB_ENCODER.output_size
            elif model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
                device = (
                    torch.device("cuda", model_config.TORCH_GPU_ID)
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                self.rgb_encoder = TorchVisionResNet50(
                    observation_space, model_config.RGB_ENCODER.output_size, device
                )
                rnn_input_size += model_config.RGB_ENCODER.output_size
            elif model_config.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
                self.rgb_encoder = ResnetRGBEncoder(
                    observation_space,
                    output_size=model_config.RGB_ENCODER.output_size,
                    backbone=model_config.RGB_ENCODER.backbone,
                    trainable=model_config.RGB_ENCODER.train_encoder,
                    normalize_visual_inputs=model_config.normalize_visual_inputs,
                    obs_augmentations=obs_augmentations,
                    resnet_baseplanes=model_config.resnet_baseplanes,
                )
                rnn_input_size += model_config.RGB_ENCODER.output_size
            else:
                self.rgb_encoder = None
        else:
            logger.info("Setting up no vision baseline")

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")
        
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        x = []

        if not self.model_config.NO_VISION:
            if self.depth_encoder is not None:
                depth_embedding = self.depth_encoder(observations)
                x.append(depth_embedding)
            
            if self.rgb_encoder is not None:
                rgb_embedding = self.rgb_encoder(observations)
                x.append(rgb_embedding)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))
        
        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class Seq2SeqModel(nn.Module):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config, obs_augmentations
    ):
        super().__init__()
        self.net = Seq2SeqNet(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
            obs_augmentations=obs_augmentations,
        )
        self.action_distribution = CategoricalNet(
            self.net.output_size, action_space.n
        )
        self.train()
    
    def forward(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:

        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        return distribution.logits, rnn_hidden_states

    # def act(
    #     self, observations, rnn_hidden_states, prev_actions, masks
    # ) -> CustomFixedCategorical:

    #     features, rnn_hidden_states = self.net(
    #         observations, rnn_hidden_states, prev_actions, masks
    #     )
    #     distribution = self.action_distribution(features)

    #     return distribution, rnn_hidden_states


# class ILPolicy(nn.Module, metaclass=abc.ABCMeta):
#     def __init__(self, net, dim_actions):
#         super().__init__()
#         self.net = net
#         self.dim_actions = dim_actions

#         self.action_distribution = CategoricalNet(
#             self.net.output_size, self.dim_actions
#         )

#     def forward(self, *x):
#         raise NotImplementedError

#     def evaluate_actions(
#         self, observations, rnn_hidden_states, prev_actions, masks
#     ):
#         features, rnn_hidden_states = self.net(
#             observations, rnn_hidden_states, prev_actions, masks
#         )
#         distribution = self.action_distribution(features)
#         return distribution.logits, rnn_hidden_states

#     @classmethod
#     @abc.abstractmethod
#     def from_config(cls, config, observation_space, action_space):
#         pass


# @baseline_registry.register_policy
# class ObjecNavILPolicy(ILPolicy):
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         action_space,
#         hidden_size: int = 512,
#         **kwargs
#     ):
#         super().__init__(
#             Seq2SeqNet(  # type: ignore
#                 observation_space=observation_space,
#                 hidden_size=hidden_size,
#                 num_actions=action_space.n,
#                 **kwargs,
#             ),
#             action_space.n,
#         )

#     @classmethod
#     def from_config(
#         cls, config: Config, observation_space: spaces.Dict, action_space
#     ):
#         return cls(
#             observation_space=observation_space,
#             action_space=action_space,
#             hidden_size=config.MODEL.hidden_size,
#         )
