from abc import ABC, abstractmethod
from typing import Tuple

import torch
import gym

from core.networks.base_actor import BaseActor
from core.networks.base_critic import BaseCritic


class BaseActorCritic(ABC):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Abstract class outlines functions expected from actor-critic implementations.

        Args:
          observation_space (gym.Space): Observation space in which the agent operates.
          action_space (gym.Space): Action space in which the agent operates.
        """
        if len(observation_space.shape) != 1:
            raise NotImplementedError("Expected vectorized 1-d observation space.")

        if action_space.__class__.__name__ not in ["Discrete", "Box"]:
            raise NotImplementedError("Expected `Discrete` or `Box` action space.")

    @abstractmethod
    def act(
        self,
        observations: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state return the action to take, log probability of said action, and the current state value
        computed by the critic.

        Args:
          observations (torch.Tensor): The state in which to take an action.
          recurrent_states (torch.Tensor): Recurrent states that are being used in memory-based policies.
          recurrent_state_masks (torch.Tensor): Masks applied to recurrent states.
          deterministic (bool): Whether to act in a deterministic way.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(
        self,
        observations: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given a state returns its corresponding value.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states (torch.Tensor): Recurrent states that are being used in memory-based policies.
          recurrent_state_masks (torch.Tensor): Masks based on terminal states.

        Returns:
          torch.Tensor
        """
        raise NotImplementedError

    def evaluate_actions(
        self,
        inputs: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given observations, encoded states, recurrent state recurrent_state_masks, actions.

        Args:
            inputs (torch.Tensor): Observations / states.
            recurrent_states (torch.Tensor): Recurrent states.
            recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.
            actions (torch.Tensor): Actions taken.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def recurrent_state_size(self) -> int:
        """
        Returns the size of the encoded state (eg. hidden state in a recurrent agent).

        Returns:
          int
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def actor(self) -> BaseActor:
        """
        Return the actor.

        Returns:
          BaseActor
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def critic(self) -> BaseCritic:
        """
        Return the critic.

        Returns:
          BaseCritic
        """
        raise NotImplementedError

    @abstractmethod
    def to_device(self, device: torch.device) -> "BaseActorCritic":
        """
        Performse device conversion on the actor and critic.

        Returns:
          BaseCritic
        """
        raise NotImplementedError

    def forward(self) -> None:
        """
        Forward pass for the network, in this case not implemented.

        Returns:
          None
        """
        raise NotImplementedError
