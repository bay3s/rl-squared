from typing import Tuple, Any
import numpy as np

import gym
from gym.utils import EzPickle

from core.envs.base_meta_env import BaseMetaEnv

from gym import spaces


class NavigationEnv(EzPickle, BaseMetaEnv):
    def __init__(
        self,
        max_episode_steps: int = 100,
        low: float = -0.5,
        high: float = 0.5,
        seed: int = None,
    ):
        """
        2D navigation problems, as described in [1].

        The code is adapted from https://github.com/cbfinn/maml_rl/

        At each time step, the 2D agent takes an action (its velocity, clipped in [-0.1, 0.1]), and receives a penalty
        equal to its L2 distance to the goal position (ie. the reward is `-distance`).

        The 2D navigation tasks are generated by sampling goal positions from the uniform distribution on [-0.5, 0.5]^2.

        [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep
        Networks", 2017 (https://arxiv.org/abs/1703.03400)
        """
        EzPickle.__init__(self)
        BaseMetaEnv.__init__(self)

        self.viewer = None
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        self._num_dimensions = 2
        self._start_state = np.zeros(self._num_dimensions)

        self._low = low
        self._high = high

        # sampled later
        self._current_state = None
        self._goal_position = None

        # spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

        # sample
        self.sample_task()
        pass

    def sample_task(self):
        """
        Sample a new goal position for the navigation task

        Returns:
            None
        """
        self._current_state = self._start_state
        self._elapsed_steps = 0

        self._goal_position = self.np_random.uniform(self._low, self._high, size=2)
        pass

    def reset(self) -> np.ndarray:
        """
        Resets the environment and returns the current observation.

        Returns:
            np.ndarray
        """
        self._current_state = self._start_state
        self._elapsed_steps = 0

        return self._current_state

    def step(self, action: np.ndarray) -> Tuple:
        """
        Take a step in the environment and return the corresponding observation, action, reward, plus additional
        info.

        Args:
            action (np.ndarray): ACtion to be taken in the environment.

        Returns:
            Tuple
        """
        action = np.clip(action, -0.1, 0.1)
        assert self.action_space.contains(action)
        self._start_state = self._start_state + action

        x = self._start_state[0] - self._goal_position[0]
        y = self._start_state[1] - self._goal_position[1]

        reward = -np.sqrt(x**2 + y**2)

        done = (np.abs(x) < 0.01) and (np.abs(y) < 0.01)
        time_exceeded = self.elapsed_steps == self.max_episode_steps

        return self._start_state, reward, (done or time_exceeded), {}

    @property
    def observation_space(self) -> gym.Space:
        """
        Returns the observation space of the environment.

        Returns:
          gym.Space
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: Any) -> None:
        """
        Set the observation space for the environment.

        Returns:
          gym.Space
        """
        self._observation_space = value

    @property
    def action_space(self) -> gym.Space:
        """
        Returns the action space

        Returns:
          gym.Space
        """
        return self._action_space

    @action_space.setter
    def action_space(self, value: Any) -> None:
        """
        Set the action space for the environment.

        Returns:
            gym.Space
        """
        self._action_space = value

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Get the observation space and the action space.

        Returns:
            Tuple
        """
        return self._observation_space, self._action_space

    def elapsed_steps(self) -> int:
        """
        Returns the elapsed number of episode steps in the environment.

        Returns:
          int
        """
        raise self._elapsed_steps

    def max_episode_steps(self) -> int:
        """
        Returns the maximum number of episode steps in the environment.

        Returns:
          int
        """
        return self._max_episode_steps

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
