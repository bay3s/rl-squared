from typing import List, Any
from abc import ABC

import numpy as np
from gym.core import Env


class MetaEnv(Env, ABC):

  def __init__(self):
    """
    Wrapper around OpenAI gym environments, interface for meta learning
    """
    super().__init__()

    self.np_random = np.random
    self._np_random = np.random
    pass

  def sample_tasks(self, n_tasks) -> List:
    """
    Samples task of the meta-environment

    Args:
        n_tasks (int) : number of different meta-tasks needed

    Returns:
        tasks (list) : an (n_tasks) length list of tasks
    """
    raise NotImplementedError

  def set_task(self, task) -> None:
    """
    Sets the specified task to the current environment

    Args:
        task: task of the meta-learning environment
    """
    raise NotImplementedError

  def get_task(self) -> Any:
    """
    Gets the task that the agent is performing in the current environment

    Returns:
        task: task of the meta-learning environment
    """
    raise NotImplementedError

  def log_diagnostics(self, paths, prefix) -> None:
    """
    Logs env-specific diagnostic information

    Args:
        paths (list) : list of all paths collected with this env during this iteration
        prefix (str) : prefix for logger
    """
    pass
