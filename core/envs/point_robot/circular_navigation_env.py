import random
from typing import Tuple
import numpy as np

import gym
from gym.utils import EzPickle

from core.envs.base_meta_env import BaseMetaEnv

from gym import spaces


class CircularNavigationEnv(EzPickle, BaseMetaEnv):

    SEMICIRCLE = 'semicircle'
    COMPLETE_CIRCLE = 'circle'

    def __init__(self, episode_length: int = 100, type: str = None, seed: int = None):
        """
        Initialize a 2D environment for a Point Robot to navigate.

        Args:
          episode_length (int): Max episode length for the navigation task.
          type (str): Type of goal sampling to use.
          seed (int): Random seed.
        """
        EzPickle.__init__(self)
        BaseMetaEnv.__init__(self, seed)

        self.viewer = None

        if type == self._sample_semi_circle:
          self._sampler = self._sample_circle
        else:
          self._sampler = self._sample_semi_circle

        self._num_dimensions = 2
        self._radius = 1.
        self._episode_length = episode_length
        self._num_steps = 0
        self._start_state = np.zeros(self._num_dimensions)

        # sampled later
        self._current_state = None
        self._goal_position = None

        # spaces
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        # sample
        self.sample_task()
        pass

    def _sample_semi_circle(self) -> np.ndarray:
      """
      Samples a goal position on the radius of the semicircle.

      Returns:
        np.ndarray
      """
      angle = self.np_random.uniform(0, np.pi)

      return self._radius * np.array((np.cos(angle), np.sin(angle)))

    def _sample_circle(self) -> np.ndarray:
      """
      Sample a goal position on the radius of a circle.

      Returns:
          np.ndarray
      """
      angle = self.np_random.uniform(0, 2 * np.pi)

      return self._radius * np.array((np.cos(angle), np.sin(angle)))

    @property
    def observation_space(self) -> gym.Space:
      """
      Returns the observation space for the environment.

      Returns:
        gym.Space
      """
      return self._observation_space

    @property
    def action_space(self) -> gym.Space:
      """
      Returns the action space for the environment.

      Returns:
        gym.Space
      """
      return self._action_space

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
      """
      Returns the observation space and the action space.

      Returns:
        Tuple[gym.Space, gym.Space]
      """
      return self._observation_space, self._action_space

    def sample_task(self) -> None:
      """
      Sample a new goal position for the navigation task

      Returns:
        None
      """
      self._current_state = self._start_state
      self._num_steps = 0

      self._goal_position = self._sampler()
      pass

    def reset(self) -> np.ndarray:
      """
      Resets the environment and returns the current observation.

      Returns:
        np.ndarray
      """
      self._current_state = self._start_state
      self._num_steps = 0

      return self._current_state

    def step(self, action: np.ndarray) -> Tuple:
        """
        Take a step in the environment and return the corresponding observation, action, reward, plus additional info.

        Args:
          action (np.ndarray): Action to be taken in the environment.

        Returns:
          Tuple
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        assert self.action_space.contains(action), action

        self._current_state = self._current_state + 0.1 * action
        reward = - np.linalg.norm(self._current_state - self._goal_position, ord=2)

        x = self._start_state[0] - self._goal_position[0]
        y = self._start_state[1] - self._goal_position[1]

        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01)) or (self._num_steps == self._episode_length)

        return self._current_state, reward, done, {}

    def render(self, mode: str = "human") -> None:
      """
      Render the environment given the render mode.

      Args:
        mode (str): Mode in which to render the environment.

      Returns:
        None
      """
      pass

    def close(self) -> None:
      """
      Close the environment.

      Returns:
        None
      """
      pass
