from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import gym


class BaseCritic(ABC, nn.Module):
    def __init__(self, observation_space: gym.Space):
        """
        Abstract class for critics - outlines required functions for critics.

        Args:
          observation_space (int): Observation space for the critic.
        """
        super(BaseCritic, self).__init__()

        if len(observation_space.shape) != 1:
            raise NotImplementedError("Expected vectorized 1-d observation space.")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.

        Returns:
          torch.Tensor
        """
        raise NotImplementedError
