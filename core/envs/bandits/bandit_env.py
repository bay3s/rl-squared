from typing import Tuple
import numpy as np

import gym
from gym.utils import EzPickle
import gym.spaces as spaces

from core.envs.base_meta_env import BaseMetaEnv


class BanditEnv(gym.Env, EzPickle, BaseMetaEnv):

    def __init__(self, num_actions: int, seed: int = None):
        """
        Initialize a multi-armed bandit.

        Args:
          num_actions (int): Number of actions that the bandit is able to take.
        """
        EzPickle.__init__(self)
        BaseMetaEnv.__init__(self, seed)

        self.seed(seed)
        self.viewer = None

        self._num_actions = num_actions
        self._state = np.array([0.0])
        self._payout_odds = None

        # spaces
        self._action_space = spaces.Discrete(num_actions)

        high = np.array([1.0], dtype=np.float32)
        self._observation_space = spaces.box.Box(-high, high)

        # sample
        self.sample_task()
        pass

    def sample_task(self) -> None:
        """
        Sample a new multi-armed bandit problem from distribution over problems.

        Returns:
          None
        """
        self._payout_odds = self.np_random.uniform(low=0.0, high=1.0, size=self._num_actions)
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

    def step(self, action: int) -> Tuple:
        """
        Take a step in the environment and return the corresponding observation, action, reward,
        additional info, etc.

        Args:
          action (int): Action to be taken in the environment.

        Returns:
          Tuple
        """
        reward = self.np_random.binomial(n=1, p=self._payout_odds[action], size=1)[0]

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
