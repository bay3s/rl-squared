from typing import Tuple

import torch
import gym

from core.networks.base_actor_critic import BaseActorCritic
from core.networks.stateful.stateful_actor import StatefulActor
from core.networks.stateful.stateful_critic import StatefulCritic

from core.networks.base_actor import BaseActor
from core.networks.base_critic import BaseCritic


class StatefulActorCritic(BaseActorCritic):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, recurrent_state_size: int):
        """
        Actor-Critic for a discrete action space.

        Args:
          observation_space (gym.Space): Observation space in which the agent operates.
          action_space (gym.Space): Action space in which the agent operates.
        """
        super(StatefulActorCritic, self).__init__(observation_space, action_space)

        self._actor = StatefulActor(
            observation_space = observation_space,
            action_space = action_space,
            recurrent_state_size = recurrent_state_size,
            hidden_sizes = [256]
        )

        self._critic = StatefulCritic(
            observation_space = observation_space,
            recurrent_state_size = recurrent_state_size,
            hidden_sizes = [256]
        )

        self._recurrent_state_size = recurrent_state_size
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
        recurrent_states_actor: torch.Tensor,
        recurrent_states_critic: torch.Tensor,
        done_masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state return the action to take, log probability of said action, and the current state value
        computed by the critic.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states_actor (torch.Tensor): Recurrent states for the actor.
          recurrent_states_critic (torch.Tensor): Recurrent states for the critic.
          deterministic (bool): Whether to choose actions deterministically.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        recurrent_masks = torch.ones(done_masks.shape)

        value_estimate, recurrent_states_critic = self.critic(observations, recurrent_states_critic, recurrent_masks)
        action_distribution, recurrent_states_actor = self.actor(observations, recurrent_states_actor, recurrent_masks)

        actions = action_distribution.mode() if deterministic else action_distribution.sample()

        return value_estimate, actions, action_distribution.log_probs(actions), recurrent_states_actor, \
            recurrent_states_critic

    def get_value(self, observations: torch.Tensor, recurrent_states_critic: torch.Tensor,
                  recurrent_masks: torch.Tensor) -> torch.Tensor:
        """
        Given a state returns its corresponding value.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states_critic (torch.Tensor): Recurrent states that are being used in memory-based policies.

        Returns:
          torch.Tensor
        """
        return self.critic(observations, recurrent_states_critic, recurrent_masks)

    def evaluate_actions(self, inputs, actions, recurrent_states_actor, recurrent_states_critic, recurrent_masks) -> Tuple:
        """
        Evaluate actions given observations, encoded states, done_masks, actions.

        Returns:
          Tuple
        """
        value, _ = self.critic(inputs, recurrent_states_critic, recurrent_masks)
        dist, _ = self.actor(inputs, recurrent_states_actor, recurrent_masks)

        log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return value, log_probs, dist_entropy

    @property
    def recurrent_state_size(self) -> int:
        """
        Returns the size of the encoded state (eg. hidden state in a recurrent agent).

        Returns:
          int
        """
        return self._recurrent_state_size

    def forward(self) -> None:
        """
        Forward pass for the network, in this case not implemented.

        Returns:
          None
        """
        raise NotImplementedError
