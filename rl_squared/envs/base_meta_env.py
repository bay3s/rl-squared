from typing import Union, Tuple, List, Any, Optional
from abc import ABC, abstractmethod

import numpy as np

import gym
from gym.utils import seeding


class BaseMetaEnv(gym.Env, ABC):
    """
    Outline expected functionality for environments being used in meta-learning experiments.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize a base meta-learning environment.

        Args:
          seed (int): Random seed.
        """
        gym.Env.__init__(self)

        self.np_random = np.random.RandomState()
        self.seed(seed)
        pass

    def sample_task(self) -> None:
        """
        Sample a new multi-armed bandit problem from distribution over problems.

        Returns:
          None
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

    @observation_space.setter
    @abstractmethod
    def observation_space(self, value: Any) -> gym.Space:
        """
        Set the observation space for the environment.

        Returns:
          gym.Space
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """
        Returns the action space for the environment.

        Returns:
            gym.Space
        """
        raise NotImplementedError

    @action_space.setter
    @abstractmethod
    def action_space(self, value: Any) -> gym.Space:
        """
        Set the action space for the environment.

        Returns:
            gym.Space
        """
        raise NotImplementedError

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Returns the observation space and the action space.

        Returns:
          Tuple[gym.Space, gym.Space]
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> List:
        """
        Set the seed for np.random

        Args:
          seed (int): Seed to set for random number generator.

        Returns:
            List
        """
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

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
        raise NotImplementedError

    @property
    def elapsed_steps(self) -> int:
        """
        Returns the elapsed number of episode steps in the environment.

        Returns:
          int
        """
        raise NotImplementedError

    @property
    def max_episode_steps(self) -> int:
        """
        Returns the maximum number of episode steps in the environment.

        Returns:
          int
        """
        raise NotImplementedError

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple:
        """
        Resets the environment and returns the corresponding observation.

        Args:
            seed (int): Random seed.
            options (dict): Additional options.

        Returns:
            Tuple
        """
        raise NotImplementedError

    def render(self, mode: str = "human") -> None:
        """
        Render the environment.

        Args:
          mode (str): Render mode.

        Returns:
          None
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the current environment.

        Returns:
          None
        """
        raise NotImplementedError
