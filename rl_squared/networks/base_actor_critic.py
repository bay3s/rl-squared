from abc import ABC, abstractmethod
from typing import Tuple

import torch
import gym

from rl_squared.networks.base_actor import BaseActor
from rl_squared.networks.base_critic import BaseCritic


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
        recurrent_states_actor: torch.Tensor,
        recurrent_states_critic: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state return the action to take, log probability of said action, and the current state value
        computed by the critic.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states_actor (torch.Tensor): Recurrent states for the actor.
          recurrent_states_critic (torch.Tensor): Recurrent states for the critic.
          recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.
          deterministic (bool): Whether to choose actions deterministically.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(
        self,
        observations: torch.Tensor,
        recurrent_states_critic: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Given a state returns its corresponding value.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states_critic (torch.Tensor): Recurrent states that are being used in memory-based critics.
          recurrent_state_masks (torch.Tensor): Masks based on terminal states.

        Returns:
          torch.Tensor
        """
        raise NotImplementedError

    def evaluate_actions(
        self,
        inputs: torch.Tensor,
        actions: torch.Tensor,
        recurrent_states_actor: torch.Tensor,
        recurrent_states_critic: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given observations, encoded states, recurrent state recurrent_state_masks, actions.

        Args:
            inputs (torch.Tensor): Inputs to the actor and the critic.
            actions (torch.Tensor): Actions taken at each timestep.
            recurrent_states_actor (torch.Tensor): Recurrent states for the actor.
            recurrent_states_critic (torch.Tensor): Recurrent states for the critic.
            recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.

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
