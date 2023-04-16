from typing import Tuple, Union

import numpy as np
import gym


class RLSquaredEnvWrapper(gym.Wrapper):

  """
  Environment wrapper that modifies the information returned while taking a step to reflect whether the episode was
  done as a result of hitting time limits.
  """

  def step(self, action: Union[int, np.ndarray]) -> Tuple:
    """
    Take one step in the current environment given the action.

    Additionally, as per the RL^2 paper this wrapper updates the observation returned to include the previous
    action, reward, and whether the episode is done.

    Args:
        action (Any): Action to be taken in the environment.

    Returns:
        Tuple
    """
    obs, rew, done, info = self.env.step(action)

    if self.action_space.__class__.__name__ == "Discrete":
      obs = np.concatenate([obs, self._one_hot_action(action), [rew], [done]])
    else:
      # @todo handle continuous spaces.
      raise NotImplementedError

    return obs, rew, done, info

  def _one_hot_action(self, action: int) -> np.array:
    """
    In the case of discrete action spaces, this returns a one-hot encoded action.

    Returns:
      np.array
    """
    encoded_action = np.zeros(self.env.action_space.n)
    encoded_action[action] = 1.

    return encoded_action

  def reset(self, **kwargs) -> Tuple:
    """
    Reset the environment.

    Args:
        **kwargs (dict): Key word arguments.

    Returns:
        Tuple
    """
    return self.env.reset(**kwargs)

