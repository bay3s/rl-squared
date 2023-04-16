from typing import Tuple, List
import numpy as np

import gym
from gym.utils import seeding, EzPickle
import gym.spaces as spaces


class MultiArmedBanditEnv(gym.Env, EzPickle):

  PAYOUT_ODDS = np.array(
    [0.6738756, 0.86932993, 0.45309433, 0.25862459, 0.83665882, 0.99294215, 0.79695253, 0.29031321, 0.83765234,
     0.16940177])

  def __init__(self, num_actions: int, seed: int = None):
    """
    Initialize a multi-armed bandit.

    Args:
      num_actions (int): Number of actions that the bandit is able to take.
    """
    EzPickle.__init__(self)
    self.seed(seed)
    self.viewer = None

    self._num_actions = num_actions
    self._state = np.array([0.])
    self._payout_odds = self.PAYOUT_ODDS

    # @todo self.sample_task()

    # same as the number of actions
    self._action_space = spaces.Discrete(num_actions)

    # observation space
    high = np.array([1.], dtype = np.float32)
    self._observation_space = spaces.box.Box(-high, high)
    pass

  def sample_task(self) -> None:
    """
    Sample a new multi-armed bandit problem from distribution over problems.

    Returns:
      None
    """
    self._payout_odds = np.random.uniform(low = 0.0, high = 1.0, size = self._num_actions)
    pass

  def reset(self) -> np.ndarray:
    """
    Resets the environment and returns the corresponding observation.

    This is different from `sample_task`, unlike the former this will not change the payout probabilities.

    Returns:
      np.ndarray
    """
    return self._state

  @property
  def observation_space(self) -> gym.Space:
    """
    Returns the observation space of the environment.

    Returns:
      gym.Space
    """
    return self._observation_space

  @property
  def action_space(self) -> gym.Space:
    """
    Returns the action space

    Returns:
      gym.Space
    """
    return self._action_space

  def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
    """
    Returns the action space

    Returns:
      Tuple[gym.Space, gym.Space]
    """
    return self.observation_space, self.action_space

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

  def step(self, action: int) -> Tuple:
    """
    Take a step in the environment and return the corresponding observation, action, reward,
    additional info, etc.

    Args:
      action (int): Action to be taken in the environment.

    Returns:
      Tuple
    """
    reward = np.random.binomial(
      n = 1, p = self._payout_odds[action], size = 1
    )[0]

    return self._state, reward, True, {}

  def render(self, mode: str = "human") -> None:
    """
    Render the environment.

    Args:
      mode (str): Render mode.

    Returns:
      None
    """
    pass

  def close(self) -> None:
    """
    Close the current environment.

    Returns:
      None
    """
    pass
