import os
import torch.nn as nn
import datetime
from typing import List

from .rl_squared_configs import RLSquaredConfigs
from src.agents import PPO


class RLSquared:

  def __init__(self, agent: PPO, sample_fn: callable, device: str, directory: str):
    """
    Implements the core RL-squared algorithm.

    Args:
      agent (nn.Module): Agent to be tuned using RL-Squared.
      sample_fn (callable): Function used to sample environments from a given distribution for meta-learning.
      device (str): Device to use for training the PyTorch modules.
      directory (str): Directory used for logging and saving models.
    """
    self.agent = agent
    self.sample = sample_fn
    self.device = device

    self.base_directory = directory
    self.logging_directory = f'{self.base_directory}/logging/'
    self.models_directory = f'{self.base_directory}/models/'
    pass

  def meta_train(self, configs: RLSquaredConfigs, plot: bool = True, verbose: bool = True) -> PPO:
    """
    Meta-training logic for RL-Squared, returns the fine-tuned agent.

    Args:
      configs (RLSquaredConfigs): Configs for the current meta-training run.
      plot (bool): Whether to plot the progress through training iterations.
      verbose (bool): Whether to output logs to the console / terminal

    Returns:
      nn.Module
    """
    env = self.sample_environment(nState=10, epLen=20)

    agent = PPO(env)

    start_time = datetime.datetime.now()

    for t in range(trials):
      env = self.sample()

      agent.env = env
      if self.agent.is_recurrent:
        agent.policy.reset_hidden()

      log_dir = os.path.join(self.log_dir, f'meta_train/Trial_{t+1}/')
      # agent train

  def meta_test(self) -> List:
    return list()
