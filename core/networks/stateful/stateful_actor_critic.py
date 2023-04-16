from typing import Tuple

import torch
import gym

from core.networks.base_actor_critic import BaseActorCritic
from core.networks.stateful.stateful_actor import StatefulActor
from core.networks.stateful.stateful_critic import StatefulCritic

from core.networks.base_actor import BaseActor
from core.networks.base_critic import BaseCritic


class StatefulActorCritic(BaseActorCritic):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Actor-Critic for a discrete action space.

        Args:
          observation_space (gym.Space): Observation space in which the agent operates.
          action_space (gym.Space): Action space in which the agent operates.
        """
        super(StatefulActorCritic, self).__init__(observation_space, action_space)

        self._actor = StatefulActor(observation_space, action_space)
        self._critic = StatefulCritic(observation_space)
        pass

    @property
    def actor(self) -> BaseActor:
        """
        Return the actor network.

        Returns:
          BaseActor
        """
        return self._actor

    @property
    def critic(self) -> BaseCritic:
        """
        Return the critic network.

        Returns:
          BaseCritic
        """
        return self._critic

    def to_device(self, device: torch.device) -> "StatefulActorCritic":
        """
        Performs device conversion on the actor and critic.

        Returns:
          StatefulActorCritic
        """
        self._actor.to(device)
        self._critic.to(device)

        return self

    def act(
        self,
        observations: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state return the action to take, log probability of said action, and the current state value
        computed by the critic.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states (torch.Tensor): Recurrent states that are being used in memory-based policies.
          recurrent_state_masks (torch.Tensor): Masks based on terminal states.
          deterministic (bool): Whether to choose actions deterministically.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        value = self.critic(observations)

        distribution, recurrent_states = self.actor(
            observations, recurrent_states, recurrent_state_masks
        )
        actions = distribution.mode() if deterministic else distribution.sample()

        return value, actions, distribution.log_probs(actions), recurrent_states

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
        return self.critic(observations)

    def evaluate_actions(self, inputs, recurrent_states, recurrent_state_masks, actions) -> Tuple:
        """
        Evaluate actions given observations, encoded states, done_masks, actions.

        Returns:
          Tuple
        """
        value = self.critic(inputs)

        dist, recurrent_states = self.actor(inputs, recurrent_states, recurrent_state_masks)
        log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return value, log_probs, dist_entropy, recurrent_states

    @property
    def recurrent_state_size(self) -> int:
        """
        Returns the size of the encoded state (eg. hidden state in a recurrent agent).

        Returns:
          int
        """
        return self.actor.recurrent_state_size

    def forward(self) -> None:
        """
        Forward pass for the network, in this case not implemented.

        Returns:
          None
        """
        raise NotImplementedError
