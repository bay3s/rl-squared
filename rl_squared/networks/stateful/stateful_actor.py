from typing import Tuple, Union, List
import gym

import torch

from rl_squared.networks.modules.distributions import Categorical, DiagonalGaussian
from rl_squared.networks.modules.memory.gru import GRU
from rl_squared.networks.base_actor import BaseActor
from rl_squared.networks.modules.distributions import (
    FixedGaussian,
    FixedCategorical,
    FixedBernoulli,
)

from rl_squared.utils.torch_utils import init_mlp


class StatefulActor(BaseActor):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        recurrent_state_size: int,
        hidden_sizes: List[int],
        shared_gru: GRU = None
    ):
        """
        Stateful actor for a discrete action space.

        Args:
            observation_space (gym.Space): State dimensions for the environment.
            action_space (gym.Space): Action space in which the agent is operating.
            recurrent_state_size (int): Size of the recurrent state.
            hidden_sizes (List[int]): Size of the hidden layers for the policy head.
            shared_gru (GRU): GRU that is shared by the policy and value function.
        """
        super(StatefulActor, self).__init__(observation_space, action_space)

        self._recurrent_state_size = recurrent_state_size

        # modules
        if shared_gru is None:
            self._gru = GRU(observation_space.shape[0], recurrent_state_size)
        else:
            self._gru = shared_gru

        self._mlp = init_mlp(recurrent_state_size, hidden_sizes)
        self._policy_head = self._init_dist(hidden_sizes[-1], action_space)
        pass

    def _init_dist(
        self, last_hidden_size: int, action_space: gym.Space
    ) -> Union[Categorical, DiagonalGaussian]:
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

    @property
    def recurrent_state_size(self) -> int:
        """
        Return the recurrent state size.

        Returns:
          int
        """
        return self._recurrent_state_size

    def forward(
        self,
        x: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
    ) -> Tuple[Union[FixedGaussian, FixedBernoulli, FixedCategorical], torch.Tensor]:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.
          recurrent_states (torch.Tensor): Recurrent states for the actor.
          recurrent_state_masks (torch.Tensor): Masks (if any) to be applied to recurrent states.

        Returns:
          Tuple[Categorical, torch.Tensor]
        """
        x, recurrent_states = self._gru(x, recurrent_states, recurrent_state_masks)
        x = self._mlp(x)
        x = self._policy_head(x)

        return x, recurrent_states
