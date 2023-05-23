from typing import Tuple, Optional

import numpy as np
from gym.utils import EzPickle

from rl_squared.envs.ant.base_ant_env import BaseAntEnv


class AntTargetPositionEnv(BaseAntEnv, EzPickle):
    def __init__(
        self,
        episode_length: int,
        min_position: float = -3.0,
        max_position: float = 3.0,
        auto_reset: bool = True,
        seed: int = None,
    ):
        """
        Ant environment with target position.

        The code is adapted from https://github.com/cbfinn/maml_rl

        The ant follows the dynamics from MuJoCo [1], and receives at each time step a reward composed of a control
        cost, a contact cost, a survival reward, and a penalty equal to its L1 distance to the target position.

        The tasks are generated by sampling the target positions from the uniform distribution on [-3, 3]^2.

        [1] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for model-based control", 2012
            (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

        Args:
            episode_length (int): Maximum number of steps before the episode is terminated.
            min_position (float): Lowerbound for target position (for x & y axis).
            max_position (float): Upperbound for target position (for x & y axis).
            auto_reset (bool): Whether to auto-reset.
            seed (int): Random seed.
        """
        self._episode_length = episode_length
        self._elapsed_steps = 0
        self._auto_reset = auto_reset
        self._episode_reward = 0.0

        self._min_position = min_position
        self._max_position = max_position

        # set a stub, sample later.
        self._target_position = np.random.uniform(
            self._min_position, self._max_position, size=2
        )

        BaseAntEnv.__init__(self)
        EzPickle.__init__(self)

        # sample
        self.seed(seed)
        self.sample_task()
        pass

    def step(self, action: np.ndarray) -> Tuple:
        """
        Take a step in the environment.

        Args:
            action (np.ndarray): Action to be taken in the environment.

        Returns:
            Tuple
        """
        self._elapsed_steps += 1

        self.do_simulation(action, self.frame_skip)
        current_position = np.array(self.get_body_com("torso"))[:2]

        goal_reward = -np.sum(np.abs(current_position - self._target_position))
        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        self._episode_reward += reward

        observation = self._get_obs()
        truncated = self.elapsed_steps == self.max_episode_steps
        terminated = truncated
        done = truncated or terminated

        info = {}
        if done and self._auto_reset:
            info["episode"] = {}
            info["r"] = self._episode_reward
            observation, _ = self.reset()
            pass

        return observation, reward, terminated, truncated, info

    def sample_task(self) -> None:
        """
        Sample a target position for the task

        Returns:
            None
        """
        self._target_position = self.np_random.uniform(
            self._min_position, self._max_position, size=2
        )

        self._elapsed_steps = 0
        self._episode_reward = 0.0
        pass

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> np.ndarray:
        """
        Reset the environment to the start state.

        Args:
            seed (int): Random seed.
            options (dict): Additional options.

        Returns:
            np.ndarray
        """
        self._elapsed_steps = 0
        self._episode_reward = 0.0

        return BaseAntEnv.reset(self, seed=seed, options=options)

    def elapsed_steps(self) -> int:
        """
        Return the elapsed steps.

        Returns:
            int
        """
        return self._elapsed_steps

    def max_episode_steps(self) -> int:
        """
        Return the maximum episode steps.

        Returns:
            int
        """
        return self._episode_length
