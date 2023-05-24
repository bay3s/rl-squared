import numpy as np
from typing import Tuple, List

import torch

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


class PyTorchVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, device: torch.device):
        """
        Initialize an environment compatible with PyTorch.

        Args:
            venv (VecEnv): Vectorized environment to provide a PyTorch wrapper for.
            device (torch.device): Device for PyTorch tensors.
        """
        super(PyTorchVecEnvWrapper, self).__init__(venv)
        self.device = device
        pass

    def reset(self, seed=None) -> torch.Tensor:
        """
        Reset the environment and retur the observation.

        Returns:
            torch.Tensor
        """
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)

        return obs

    def step_async(self, actions: torch.Tensor) -> None:
        """
        Aysnc step in the vectorized environment.

        Args:
            actions (torch.Tensor): Tensor containing actions to be taken in the environment(s).

        Returns:
            None
        """
        actions = actions.cpu()

        if isinstance(actions, torch.LongTensor):
            # squeeze dimensions for discrete actions
            actions = actions.squeeze(1)

        actions = actions.cpu().numpy()
        self.venv.step_async(actions)
        pass

    def step_wait(self) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List]:
        """
        Wait for the step taken with step_async() and return resulting observations, rewards, etc.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, reward, done, info
