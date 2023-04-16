from typing import Tuple, Union
import gym
import numpy as np

import torch.nn as nn
import torch

from core.networks.modules.distributions import Categorical, DiagonalGaussian
from core.networks.base_actor import BaseActor
from core.utils.torch_utils import init_module


class StatefulActor(BaseActor):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_size: int = 64,
    ):
        """
        Stateful actor for a discrete action space.

        Args:
          observation_space (gym.Space): State dimensions for the environment.
          action_space (gym.Space): Action space in which the agent is operating.
        """
        super(StatefulActor, self).__init__(observation_space, action_space)

        self._hidden_size = hidden_size

        self._gru = nn.GRU(observation_space.shape[0], hidden_size)
        for name, param in self._gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        _init = lambda m: init_module(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self._mlp = nn.Sequential(
            _init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        if action_space.__class__.__name__ == "Discrete":
            self._dist = Categorical(hidden_size, action_space.n)
        elif action_space.__class__.__name__ == "Box":
            self._dist = DiagonalGaussian(hidden_size, action_space.shape[0])

    @property
    def recurrent_state_size(self) -> int:
        """
        Return the recurrent state size.

        Returns:
          int
        """
        return self._hidden_size

    def forward(
        self, x: torch.Tensor, recurrent_states: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[Union[Categorical, DiagonalGaussian], torch.Tensor]:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.
          recurrent_states (torch.Tensor): Recurrent states for the actor.

        Returns:
          Tuple[Categorical, torch.Tensor]
        """
        x, recurrent_states = self._forward_gru(x, recurrent_states, masks)
        x = self._mlp(x)

        return self._dist(x), recurrent_states

    def _forward_gru(self, x, recurrent_states: torch.Tensor, recurrent_state_masks: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the GRU unit.

        Args:
            x (torch.Tensor): Input to the GRU.
            recurrent_states (torch.Tensor): Recurrent state from the previous forward pass.
            recurrent_state_masks (torch.Tensor): Masks for the recurrent state.

        Returns:
            Tuple
        """
        if x.size(0) == recurrent_states.size(0):
            # @todo this line resets the recurrent state, should be changed for RL-Squared.
            x, recurrent_states = self._gru(
                x.unsqueeze(0), (recurrent_states * recurrent_state_masks).unsqueeze(0)
            )
            x = x.squeeze(0)
            recurrent_states = recurrent_states.squeeze(0)

            return x, recurrent_states

        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = recurrent_states.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        # Same deal with done_masks
        recurrent_state_masks = recurrent_state_masks.view(T, N)

        # Let's figure out which steps in the sequence have a zero for any agent
        # We will always assume t=0 has a zero in it as that makes the logic cleaner
        has_zeros = (recurrent_state_masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the done_masks[1:]
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [T]

        recurrent_states = recurrent_states.unsqueeze(0)
        outputs = []

        for i in range(len(has_zeros) - 1):
            # We can now process steps that don't have any zeros in done_masks together!
            # This is much faster
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, recurrent_states = self._gru(
                x[start_idx:end_idx],
                recurrent_states * recurrent_state_masks[start_idx].view(1, -1, 1),
            )

            outputs.append(rnn_scores)
            pass

        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)

        # flatten
        x = x.view(T * N, -1)
        recurrent_states = recurrent_states.squeeze(0)
        pass

        return x, recurrent_states
