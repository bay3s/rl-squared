from typing import Tuple, Any
from abc import ABC

import numpy as np
import gym

from gym.envs.mujoco import AntEnv as AntEnv_
from rl_squared.envs.base_meta_env import BaseMetaEnv


class BaseAntEnv(AntEnv_, BaseMetaEnv, ABC):
    def __init__(self):
        """
        Initialize the Mujoco Ant environment for meta-learning.
        """
        self._action_scaling = None

        BaseMetaEnv.__init__(self)
        AntEnv_.__init__(self)
        pass

    @property
    def action_scaling(self) -> float:
        """
        Scale the action space.

        Returns:
            float
        """
        if (not hasattr(self, "action_space")) or (self.action_space is None):
            return 1.0

        if self._action_scaling is None:
            lb, ub = self.action_space.low, self.action_space.high
            self._action_scaling = 0.5 * (ub - lb)

        return self._action_scaling

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Returns the observation space and the action space.

        Returns:
            Tuple
        """
        return self.observation_space, self.action_space

    @property
    def observation_space(self) -> gym.Space:
        """
        Returns the observation space for the environment.

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
        Set the action space for the environment.

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

    def viewer_setup(self) -> None:
        """
        Set up the viewer for rendering the environment.

        Returns:
            None
        """
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True

    def render(self, mode: str = "human"):
        """
        Render the enevironment.

        Args:
            mode (str): Mode in which to render the environment.

        Returns:
            None
        """
        if mode == "rgb_array":
            self._get_viewer(mode).render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return data
        elif mode == "human":
            self._get_viewer(mode).render()
