import torch
import torch.nn as nn

from core.networks.modules.distributions.bernoulli.fixed_bernoulli import FixedBernoulli
from core.utils.torch_utils import init_module


class Bernoulli(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initialize the Bernoulli distribution with a single linear layer.

        Args:
            num_inputs (int): Number of inputs.
            num_outputs (int): Number of outputs.
        """
        super(Bernoulli, self).__init__()

        self.linear = init_module(
            nn.Linear(num_inputs, num_outputs),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
        )

    def forward(self, x) -> FixedBernoulli:
        """
        Forward pass for the Bernoulli distribution.

        Args:
            x (torch.Tensor):

        Returns:
            FixedBernoulli
        """
        x = self.linear(x)

        return FixedBernoulli(logits=x)
