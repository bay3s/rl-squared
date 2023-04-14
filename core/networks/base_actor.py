from typing import Union
from abc import ABC, abstractmethod

import gym
import torch
import torch.nn as nn


from core.networks.modules.distributions import FixedGaussian, FixedCategorical, FixedBernoulli


class BaseActor(ABC, nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Actor-Critic for a discrete action space.

        Args:
          observation_space (gym.Space): State dimensions for the environment.
          action_space (gym.Space): Action space in which the agent is operating.
        """
        super(BaseActor, self).__init__()

        if len(observation_space.shape) != 1:
            raise NotImplementedError("Expected vectorized 1-d observation space.")

        if action_space.__class__.__name__ not in ["Discrete", "Box"]:
            raise NotImplementedError("Expected `Discrete` or `Box` action space.")
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> Union[FixedGaussian, FixedBernoulli, FixedCategorical]:
        """
        Forward pass through the network and return a distribution.

        Args:
          x (torch.Tensor): Input for the forward pass.

        Returns:
          Union[FixedGaussian, FixedBernoulli, FixedCategorical]
        """
        raise NotImplementedError
