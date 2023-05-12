import torch

from rl_squared.utils.torch_utils import init_module
import torch.nn as nn

from rl_squared.networks.modules.distributions.categorical.fixed_categorical import (
    FixedCategorical,
)


class Categorical(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initialize the categorical distribution.

        Args:
            num_inputs (int): Number of inputs.
            num_outputs (int): Number of outputs.
        """
        super(Categorical, self).__init__()

        self.linear = init_module(
            nn.Linear(num_inputs, num_outputs),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01,
        )
        pass

    def forward(self, x: torch.Tensor) -> FixedCategorical:
        """
        Forward pass for the Categorical distribution.

        Args:
            x (torch.Tensor): Input for the forward pass.

        Returns:
            FixedCategorical
        """
        x = self.linear(x)

        return FixedCategorical(logits=x)
