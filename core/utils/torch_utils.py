from typing import Callable, List

import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils import weight_norm


def init_gru(input_size: int, recurrent_state_size: int) -> nn.Module:
    """
    Initialize a GRU module.

    Args:
        input_size (int): Input size to the GRU.
        recurrent_state_size (int): Recurrent state size for the GRU.

    Returns:
        nn.Module
    """
    gru = nn.GRU(input_size, recurrent_state_size)

    for name, param in gru.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param)

    return gru


def init_module(
    module: nn.Module, weight_init: Callable, bias_init: Callable, gain: float = 1.0
) -> nn.Module:
    """
    Initialize a module with the given weight and bias functions.

    Args:
        module (nn.Module): Module that is to be initialized with the given weight and bias.
        weight_init (Callable): Function for initializing weights.
        bias_init (Callable): Function for initialize biases.
        gain (float): Gain amount.

    Returns:
        nn.Module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    weight_norm(module)

    return module


def init_mlp(input_size: int, hidden_sizes: List[int]) -> nn.Sequential:
    """
    Initialize the value head for the critic.

    Args:
        input_size (List[int]): Size of the recurrent state in the base RNN.
        hidden_sizes (List[int]): Sizes of the hidden layers of the MLP.

    Returns:
        nn.Sequential
    """

    def _init_orthogonal(m: nn.Module):
        return init_module(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

    feature_sizes = list([input_size])
    feature_sizes.extend(hidden_sizes)

    mlp_modules = list()
    for i in range(len(feature_sizes) - 1):
        hidden_layer = _init_orthogonal(
            nn.Linear(feature_sizes[i], feature_sizes[i + 1])
        )

        # zero bias
        torch.nn.init.zeros_(hidden_layer.bias)
        mlp_modules.append(hidden_layer)

        # relu
        mlp_modules.append(nn.ReLU())
        pass

    return nn.Sequential(*mlp_modules)
