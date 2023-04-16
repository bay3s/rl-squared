from typing import List, Tuple
from abc import ABC, abstractmethod

import gym


class BaseMetaEnv(ABC):
  """
  Abstract class that outlines functions required by an environment for meta-learning.
  """

  @property
  @abstractmethod
  def sample_task(self) -> gym.Space:
    """
    Samples a new task for the environment.

    Returns:
      gym.Space
    """
    raise NotImplementedError

  @property
  @abstractmethod
  def observation_space(self) -> gym.Space:
    """
    Returns the observation space for the environment.

    Returns:
      gym.Space
    """
    raise NotImplementedError

  @property
  def action_space(self) -> gym.Space:
    """
    Returns the action space for the environment.

    Returns:
      gym.Space
    """
    raise NotImplementedError

  @property
  def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
    """
    Returns the observation space followed by the action space.

    Returns:
      Tuple[gym.Space, gym.Space]
    """
    raise NotImplementedError

  def seed(self, seed: int = None) -> List:
    """
    Set the seed for np.random

    Args:
      seed (int): Seed to set for random number generator.

    Returns:
      List
    """
    raise NotImplementedError


