import torch
import torch.nn as nn

from rl_squared.utils.torch_utils import init_module
from rl_squared.networks.modules.distributions.gaussian.fixed_gaussian import (
    FixedGaussian,
)


class _AddBias(nn.Module):
    def __init__(self, bias: torch.Tensor):
        """
        Class used to add bias to

        Args:
            bias (torch.Tensor): Bias to be added.
        """
        super(_AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class DiagonalGaussian(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initialize the diagonal gaussian.

        Args:
            num_inputs (int): Number of inputs.
            num_outputs (int): Number of outputs.
        """
        super(DiagonalGaussian, self).__init__()

        self.fc_mean = init_module(
            nn.Linear(num_inputs, num_outputs),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
        )

        self.logstd = _AddBias(torch.zeros(num_outputs))

    def forward(self, x: torch.Tensor) -> FixedGaussian:
        """
        Forward pass for the diagonal Gaussian.

        Args:
            x (torch.Tensor): Input for the forward pass.

        Returns:
            FixedGaussian
        """
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())

        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)

        return FixedGaussian(action_mean, action_logstd.exp())
