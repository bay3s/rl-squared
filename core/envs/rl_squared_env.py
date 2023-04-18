from typing import Union, Tuple

from abc import abstractmethod

import numpy as np
import gym
from copy import deepcopy

from core.envs.base_meta_env import BaseMetaEnv


class RLSquaredEnv:

  def __init__(self, env: BaseMetaEnv):
    """
    Abstract class that outlines functions required by an environment for meta-learning via RL-Squared.

    Args:
      env (BaseMetaEnv): Environment for general meta-learning around which to add an RL-Squared wrapper.
    """
    self._wrapped_env = env

    self._action_space = self._wrapped_env.action_space
    self._observation_space = self._make_observation_space()

    assert isinstance(self._observation_space, type(self._wrapped_env.observation_space))
    pass

  @property
  def spec(self):
    return self._wrapped_env.spec

  def _make_observation_space(self) -> gym.Space:
    """
    Modify the observation space of the wrapped environment to account for rolling forward rewards, actions, terminal
    states.

    Returns:
      gym.Space
    """
    obs_dims = gym.spaces.flatdim(self._wrapped_env.observation_space)
    action_dims = gym.spaces.flatdim(self._wrapped_env.action_space)
    new_obs_dims = obs_dims + action_dims + 2
    obs_shape = (new_obs_dims,)

    obs_space = deepcopy(self._wrapped_env.observation_space)
    obs_space._shape = obs_shape

    return obs_space

  def reset(self) -> np.ndarray:
    """
    Reset the environment and return a starting observation.

    Returns:
      np.ndarray
    """
    obs = self._wrapped_env.reset()
    obs = self._format_observation(obs, None, 0., False)

    return obs

  def step(self, action: Union[int, np.ndarray]) -> Tuple:
    """
    Take a step in the environment.

    Args:
      action (Union[int, np.ndarray]): Action to take in the environment.

    Returns:
      Tuple
    """
    wrapped_step = self._wrapped_env.step(action)
    obs, rew, done, info = wrapped_step
    next_obs = self._format_observation(obs, action, rew, done)

    return next_obs, rew, done, info

  def _format_observation(self, obs: np.ndarray, action: Union[int, np.ndarray], rew: float, done: bool) -> np.ndarray:
    """
    Given an observation, action, reward, and whether an episode is done - return the formatted observation.

    Args:
        obs (np.ndarray): Observation made.
        action (Union[int, np.ndarray]): Action taken in the state.
        rew (float): Reward received.
        done (bool): Whether this is the terminal observation.

    Returns:
        np.ndarray
    """
    if self._wrapped_env.action_space.__class__.__name__ == "Discrete":
      obs = np.concatenate([obs, self._one_hot_action(action), [rew], [float(done)]])
    else:
      # @todo continuous action spaces
      raise NotImplementedError

    return obs

  def _one_hot_action(self, action: int = None) -> np.array:
    """
    In the case of discrete action spaces, this returns a one-hot encoded action.

    Returns:
      np.array
    """
    encoded_action = np.zeros(self.action_space.n)

    if action is not None:
      encoded_action[action] = 1.0

    return encoded_action

  @property
  def observation_space(self) -> gym.Space:
    """
    Return the dimension of the observations.

    Returns:
      Union[Tuple, int]
    """
    return self._observation_space

  @property
  def action_space(self) -> gym.Space:
    """
    Return the dimension of the observations.

    Returns:
      int
    """
    return self._wrapped_env.action_space

  def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
    """
      Returns the action space

      Returns:
        Tuple[gym.Space, gym.Space]
      """
    return self.observation_space, self.action_space

  def sample_task(self) -> None:
    """
    Samples a new task for the environment.

    Returns:
      np.ndarray
    """
    self._wrapped_env.sample_task()
