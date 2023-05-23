from typing import Tuple, Any, Optional
import numpy as np

import gym
from gym.utils import EzPickle, seeding
import gym.spaces as spaces

from rl_squared.envs.base_meta_env import BaseMetaEnv


class TabularMDPEnv(EzPickle, BaseMetaEnv):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        episode_length: int,
        auto_reset: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize a tabular MDP.

        Args:
          num_states (int): Number of states.
          num_actions (int): Number of actions.
          episode_length (int): Maximum steps per episode.
          auto_reset (bool): Whether to auto-reset at end of episode.
          seed (int): Random seed.
        """
        EzPickle.__init__(self)
        BaseMetaEnv.__init__(self, seed)

        self.viewer = None
        self._episode_length = episode_length
        self._auto_reset = auto_reset

        self._elapsed_steps = 0
        self._episode_reward = 0.0

        self._num_states = num_states
        self._num_actions = num_actions
        self._start_state = 0

        # sampled later
        self._current_state = None
        self._transitions = None
        self._rewards_mean = None

        # spaces
        self.action_space = spaces.Discrete(self._num_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._num_states,)
        )

        # sample
        self.sample_task()
        pass

    def sample_task(self) -> None:
        """
        Sample a new multi-armed bandit problem from distribution over problems.

        Returns:
          None
        """
        self._current_state = self._start_state
        self._elapsed_steps = 0
        self._episode_reward = 0.0

        self._transitions = self.np_random.dirichlet(
            alpha=np.ones(self._num_states), size=(self._num_states, self._num_actions)
        )

        self._rewards_mean = self.np_random.normal(
            loc=1.0, scale=1.0, size=(self._num_states, self._num_actions)
        )

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
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self._current_state = self._start_state
        self._elapsed_steps = 0
        self._episode_reward = 0.0

        observation = np.zeros(self._num_states)
        observation[self._start_state] = 1.0

        return observation, {}

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
        Returns the action space

        Returns:
          Tuple[gym.Space, gym.Space]
        """
        return self.observation_space, self.action_space

    def step(self, action: int) -> Tuple:
        """
        Take a step in the environment and return the corresponding observation, action, reward, additional info, etc.

        Args:
            action (int): Action to be taken in the environment.

        Returns:
            Tuple
        """
        self._elapsed_steps += 1

        reward = self.np_random.normal(
            loc=self._rewards_mean[self._current_state, action], scale=1.0
        )

        self._episode_reward += reward

        try:
            self._current_state = self.np_random.choice(
                a=self._num_states, p=self._transitions[self._current_state, action]
            )
        except ValueError:
            print(self._transitions[self._current_state, action].ndim)
            print(self._transitions[self._current_state, action])
            print(self._transitions)
            raise ValueError

        observation = np.zeros(self._num_states)
        observation[self._current_state] = 1.0

        terminated = False
        truncated = self.elapsed_steps == self.max_episode_steps
        done = terminated or truncated

        info = {}
        if done and self._auto_reset:
            info["episode"] = {}
            info["episode"]["r"] = self._episode_reward
            observation, _ = self.reset()

        return observation, reward, terminated, truncated, info

    @property
    def elapsed_steps(self) -> int:
        """
        Returns the elapsed number of episode steps in the environment.

        Returns:
          int
        """
        return self._elapsed_steps

    @property
    def max_episode_steps(self) -> int:
        """
        Returns the maximum number of episode steps in the environment.

        Returns:
          int
        """
        return self._episode_length

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
