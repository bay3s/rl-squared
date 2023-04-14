from typing import Any
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize as VecNormalize_


class NormalizedVecEnv(VecNormalize_):

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Wrapper to normalize observations from a vectorized environment.

        Args:
            *args (Any): Args for the parent class.
            **kwargs (Any): Kwargs for the parent class.
        """
        super(NormalizedVecEnv, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs: np.ndarray, update: bool = True) -> bool:
        """
        Normalize observations.

        Args:
            obs (np.ndarray):
            update (bool): Update the running

        Returns:
            np.ndarray
        """
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)

            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )

            return obs
        else:
            return obs

    def train(self) -> None:
        """
        Set `training` to True.

        Returns:
            None
        """
        self.training = True

    def eval(self) -> None:
        """
        Set `training` to False.

        Returns:
            None
        """
        self.training = False
