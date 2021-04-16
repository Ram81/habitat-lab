import math
import sys
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat.tasks.nav.nav import (
    EpisodicGPSSensor,
    EpisodicCompassSensor
)
from habitat.tasks.rearrangement.rearrangement import (
    AllObjectPositions
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
from habitat_baselines.utils.common import CategoricalNet, CustomFixedCategorical


class DiscriminatorNet(Net):
    r"""A baseline discriminato network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.model_config = model_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
            self.depth_encoder = SimpleDepthCNN(
                observation_space, model_config.DEPTH_ENCODER.output_size
            )
        elif model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
                trainable=model_config.DEPTH_ENCODER.trainable,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
            "ResnetRGBEncoder",
        ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

        if model_config.RGB_ENCODER.cnn_type == "SimpleRGBCNN":
            self.rgb_encoder = SimpleRGBCNN(
                observation_space, model_config.RGB_ENCODER.output_size
            )
        elif model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, model_config.RGB_ENCODER.output_size, device
            )
        elif model_config.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
            self.rgb_encoder = ResnetRGBEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                backbone=model_config.RGB_ENCODER.backbone,
                trainable=model_config.RGB_ENCODER.train_encoder,
            )

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind
    
    @property
    def num_recurrent_layers(self):
        return 0

    def forward(self, observations):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        features = torch.cat([instruction_embedding, depth_embedding, rgb_embedding], dim=1)
        return features


class SeqDiscriminatorNet(Net):
    r"""A baseline sequential discriminator network that encodes agent, object and receptacle,
    states and passes it through an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.model_config = model_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)

        rnn_input_size = 0
        rnn_input_size += self.instruction_encoder.output_size

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
        
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        self.object_state_input_dim = observation_space.spaces[
            AllObjectPositions.cls_uuid
        ].shape[0] + 1
        self.object_state_embedding_dim = 32
        self.object_state_encoder = nn.Linear(
            self.object_state_input_dim, self.object_state_embedding_dim
        )
        # Input will be embeddings of agent, object and receptacle concatenated
        # As we have 2 objects the input size will be 2 x embedding dim
        rnn_input_size += model_config.max_objects * self.object_state_embedding_dim

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.output_dim = rnn_input_size

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )
        self._num_recurrent_layers = model_config.STATE_ENCODER.num_recurrent_layers
        self._hidden_size = model_config.STATE_ENCODER.hidden_size

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size # self.output_dim

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind
    
    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers
    
    @property
    def hidden_size(self):
        return self._hidden_size
    
    def get_object_state_encoding(self, observations):
        goal_observations = observations[AllObjectPositions.cls_uuid]
        goal_observations = torch.stack(
            [
                goal_observations[:, :, 0],
                torch.cos(-goal_observations[:, :, 1]),
                torch.sin(-goal_observations[:, :, 1]),
            ],
            -1,
        ).float()

        return self.object_state_encoder(goal_observations)

    def forward(self, observations, rnn_hidden_states, masks, prev_actions):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        x = []

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )
        
        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )
        
        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)
        # Object state embeddings
        object_state_embedding = self.get_object_state_encoding(observations)
        object_state_embedding = object_state_embedding.flatten(1)
        x.append(object_state_embedding)

        instruction_embedding = self.instruction_encoder(observations)
        x.append(instruction_embedding)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states


class DiscriminatorModel(nn.Module):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__()
        if model_config.SEQUENTIAL:
            self.net = SeqDiscriminatorNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            )
        else:
            self.net = DiscriminatorNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            )
        self.linear = nn.Linear(
            self.net.output_size, 1
        )
        self.train()
    
    def forward(
        self, observations, rnn_hidden_states, masks, prev_actions
    ) -> CustomFixedCategorical:
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, masks, prev_actions)

        logits = self.linear(features)
        return logits, rnn_hidden_states