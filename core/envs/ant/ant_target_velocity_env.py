from typing import Tuple

import numpy as np

import gym
from gym.utils import EzPickle

from core.envs.ant.base_ant_env import BaseAntEnv


class AntTargetVelocityEnv(BaseAntEnv, EzPickle):
    def __init__(
        self,
        max_episode_steps: int,
        min_velocity: float = 0.0,
        max_velocity: float = 3.0,
        seed: int = None,
    ):
        """
        Ant environment with target velocity, as described in [1].

        The code is adapted from https://github.com/cbfinn/maml_rl

        The ant follows the dynamics from MuJoCo [2], and receives at each time step a reward composed of a control
        cost, a contact cost, a survival reward, and a penalty equal to the difference between its current velocity
        and the target velocity.

        The tasks are generated by sampling the target velocities from the uniform distribution on [0, 3].

        [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep
            Networks", 2017 (https://arxiv.org/abs/1703.03400)
        [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for model-based control", 2012
            (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

        Args:
            max_episode_steps (int): Maximum number of steps per episode.
            min_velocity (float): Minimum target velocity.
            max_velocity (float): Maximum target velocity.
            seed (int): Random seed.
        """
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        self._min_velocity = min_velocity
        self._max_velocity = max_velocity

        # set a stub, sample later.
        self._target_velocity = np.random.uniform(
            self._min_velocity, self._max_velocity, size=1
        )

        BaseAntEnv.__init__(self)
        EzPickle.__init__(self)

        # sample
        self.seed(seed)
        self.sample_task()
        pass

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Returns the action space

        Returns:
          Tuple[gym.Space, gym.Space]
        """
        return self.observation_space, self.action_space

    def step(self, action: np.ndarray) -> Tuple:
        """
        Take a step in the environment and return the corresponding observation, action, reward,
        additional info, etc.

        Args:
          action (np.ndarray): Action to be taken in the environment.

        Returns:
          Tuple
        """
        self._elapsed_steps += 1
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * np.abs(forward_vel - self._target_velocity.item()) + 1.0
        survive_reward = 0.05

        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / self.action_scaling))
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()

        not_done = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = self.max_episode_steps == self.elapsed_steps and (not not_done)

        infos = dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self._target_velocity,
        )

        time_exceeded = self.elapsed_steps == self.max_episode_steps

        return observation, reward, (done or time_exceeded), infos

    def sample_task(self):
        """
        Sample a new target velocity.

        Returns:
          None
        """
        self._target_velocity = self.np_random.uniform(
            self._min_velocity, self._max_velocity, size=1
        )
        self._elapsed_steps = 0
        pass

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the start state.

        Returns:
            np.ndarray
        """
        self._elapsed_steps = 0

        return BaseAntEnv.reset(self)

    @property
    def elapsed_steps(self) -> int:
        """
        Return the elapsed steps.

        Returns:
            int
        """
        return self._elapsed_steps

    @property
    def max_episode_steps(self) -> int:
        """
        Return the maximum episode steps.

        Returns:
            int
        """
        return self._max_episode_steps