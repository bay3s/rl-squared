"""
Implementation of a basic RL environment.

Rewards are normal, transitions are multinomial.

References:
  - https://github.com/iosband/TabulaRL/blob/master/src/environment.py

Original Author:
  - iosband@stanford.edu
"""


import random
from abc import ABC
from typing import Tuple

import numpy as np


class Environment(ABC):

  def __init__(self):
    """
    Intiialize a base RL environment.
    """
    pass

  def reset(self) -> None:
    """
    Resets the environment to its starting state.

    Returns:
      None
    """
    raise NotImplementedError

  def advance(self, action: int) -> Tuple[float, int, bool]:
    """
    Advances the agent given the action and returns a 3-tuple containing the reward, new state, and whether the episode
    is done.

    Args:
      action (int): The action to take within the environment.

    Returns:
      Tuple[float, int, bool]
    """
    raise NotImplementedError


