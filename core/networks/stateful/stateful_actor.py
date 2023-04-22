from typing import Tuple, Union, List, Callable, Any
import gym
import numpy as np

import torch.nn as nn
import torch
from torch.nn import Module

from core.networks.modules.distributions import Categorical, DiagonalGaussian
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
        self._gru = self._init_recurrent(observation_space.shape[0], recurrent_state_size)
        self._mlp = self._init_mlp(recurrent_state_size, hidden_sizes)
        self._dist = self._init_dist(hidden_sizes[-1], action_space)
        pass

    def _init_recurrent(self, input_size: int, recurrent_state_size: int) -> nn.Module:
        """
        Initialize the recurrent module.

        Args:
            input_size (int): Size of the input to the recurrent layer.
            recurrent_state_size (int): Size of the recurrent state.

        Returns:
            nn.Module
        """
        gru_module = nn.GRU(input_size, recurrent_state_size)

        for name, param in gru_module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

        return gru_module

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

    def forward(self, x: torch.Tensor, recurrent_states: torch.Tensor, recurrent_masks: torch.Tensor
                ) -> Tuple[Union[Categorical, DiagonalGaussian], torch.Tensor]:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.
          recurrent_states (torch.Tensor): Recurrent states for the actor.

        Returns:
          Tuple[Categorical, torch.Tensor]
        """
        x, recurrent_states = self._forward_gru(x, recurrent_states, recurrent_masks)
        x = self._mlp(x)

        return self._dist(x), recurrent_states

    def _forward_gru(self, x, recurrent_states: torch.Tensor, recurrent_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the GRU unit.

        Args:
            x (torch.Tensor): Input to the GRU.
            recurrent_states (torch.Tensor): Recurrent state from the previous forward pass.

        Returns:
            Tuple
        """
        recurrent_state_masks = torch.ones(recurrent_masks.shape)

        if x.size(0) == recurrent_states.size(0):
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

        # masks
        recurrent_state_masks = recurrent_state_masks.view(T, N)
        has_zeros = (recurrent_state_masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the recurrent_masks[1:] where zeros are present.
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
