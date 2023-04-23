from typing import Tuple, Union, List, Callable, Any
import gym
import numpy as np

import torch.nn as nn
import torch

from core.networks.modules.distributions import Categorical, DiagonalGaussian
from core.networks.modules.memory.gru import GRU
from core.networks.base_actor import BaseActor
from core.utils.torch_utils import init_module


class StatefulActor(BaseActor):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        recurrent_state_size: int,
        hidden_sizes: List[int]
    ):
        """
        Stateful actor for a discrete action space.

        Args:
            observation_space (gym.Space): State dimensions for the environment.
            action_space (gym.Space): Action space in which the agent is operating.
            recurrent_state_size (int): Size of the recurrent state.
            hidden_sizes (List[int]): Sze of the
        """
        super(StatefulActor, self).__init__(observation_space, action_space)

        self._recurrent_state_size = recurrent_state_size

        # modules
        self._gru = GRU(observation_space.shape[0], recurrent_state_size)
        self._mlp = self._init_mlp(recurrent_state_size, hidden_sizes)
        self._dist = self._init_dist(hidden_sizes[-1], action_space)
        pass

    def _init_dist(self, last_hidden_size: int, action_space: gym.Space) -> Union[Categorical, DiagonalGaussian]:
        """
        Initialize the action distribution.

        Args:
            last_hidden_size (int): Size of the last hidden layer in the MLP.
            action_space (gym.Space): Action space for the actor.

        Returns:
            Union[Categorical, DiagonalGaussian]
        """
        if action_space.__class__.__name__ == "Discrete":
            return Categorical(last_hidden_size, action_space.n)
        elif action_space.__class__.__name__ == "Box":
            return DiagonalGaussian(last_hidden_size, action_space.shape[0])
        else:
            raise NotImplementedError

    def _init_mlp(self, recurrent_state_size: int, hidden_sizes: List[int]) -> nn.Module:
        """
        Initialize an MLP layer for the actor.

        Args:
            recurrent_state_size (int): Size of the recurrent state.
            hidden_sizes (List[int]): Sizes of the hidden layers.

        Returns:
            nn.Sequential
        """
        _init_orthogonal = lambda m: init_module(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        _init_xavier = lambda m: init_module(
            m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        feature_sizes = list([recurrent_state_size])
        feature_sizes.extend(hidden_sizes)

        mlp_modules = list()
        for i in range(len(feature_sizes) - 1):
            if i == 0:
                hidden_layer = _init_xavier(nn.Linear(feature_sizes[i], feature_sizes[i + 1]))
            else:
                hidden_layer = _init_orthogonal(nn.Linear(feature_sizes[i], feature_sizes[i + 1]))

            # zero bias
            torch.nn.init.zeros_(hidden_layer.bias)
            mlp_modules.append(hidden_layer)

            # relu
            mlp_modules.append(nn.ReLU())
            pass

        # output
        output_layer = _init_xavier(nn.Linear(feature_sizes[-1], hidden_sizes[-1]))
        torch.nn.init.zeros_(output_layer.bias)
        mlp_modules.append(output_layer)

        return nn.Sequential(*mlp_modules)

    @property
    def recurrent_state_size(self) -> int:
        """
        Return the recurrent state size.

        Returns:
          int
        """
        return self._recurrent_state_size

    def forward(self, x: torch.Tensor, recurrent_states: torch.Tensor, recurrent_masks: torch.Tensor) -> Tuple[Union[Categorical, DiagonalGaussian], torch.Tensor]:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.
          recurrent_states (torch.Tensor): Recurrent states for the actor.
          recurrent_masks (torch.Tensor): Masks (if any) to be applied to recurrent states.

        Returns:
          Tuple[Categorical, torch.Tensor]
        """
        x, recurrent_states = self._gru(x, recurrent_states, recurrent_masks)
        x = self._mlp(x)

        return self._dist(x), recurrent_states

