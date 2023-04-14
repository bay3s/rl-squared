from typing import Callable

from torch import nn as nn


def init_module(module: nn.Module, weight_init: Callable, bias_init: Callable, gain: float = 1.0):
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

    return module
