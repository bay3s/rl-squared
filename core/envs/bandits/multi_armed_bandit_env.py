from typing import Tuple, List
import numpy as np

import gym
from gym.utils import colorize, seeding, EzPickle
import gym.spaces as spaces


class MultiArmedBanditEnv(gym.Env, EzPickle):

  def __init__(self, num_actions: int):
    """
    Initialize a multi-armed bandit.

    Args:
      num_actions (int): Number of actions that the bandit is able to take.
    """
    EzPickle.__init__(self)
    self.seed()
    self.viewer = None

    # @todo action space and observation space
    self.action_space = spaces.Discrete(num_actions)
    self.observation_space = spaces.Box(-high, high)
    pass

  def seed(self, seed: int = None) -> List:
    """
    Set the seed for np.random

    Args:
      seed (int): Seed to set for random number generator.

    Returns:
      List
    """
    self.np_random, seed = seeding.np_random(seed)

    return [seed]

  @property
  def _max_episode_steps(self) -> int:
    """
    Returns the maximum episode length for this environment.

    Returns:
      int
    """
    return 1

  @property
  def _elapsed_steps(self) -> int:
    """
    Returns the maximum episode length for this environment.

    Returns:
      int
    """
    return 1


