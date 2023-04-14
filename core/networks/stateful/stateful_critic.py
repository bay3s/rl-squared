import torch.nn as nn
import torch

import gym

from core.networks.base_critic import BaseCritic


class StatefulCritic(BaseCritic):
    def __init__(self, observation_space: gym.Space):
        """
        Stateful critic.

        Args:
          observation_space (int): Observation space for the critic.
        """
        super(StatefulCritic, self).__init__(observation_space)

        if len(observation_space.shape) != 1:
            raise NotImplementedError("Expected vectorized 1-d observation space.")

        self._module = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.

        Returns:
          torch.Tensor
        """
        return self._module(x)
