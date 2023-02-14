from abc import ABC, abstractmethod
from src.environments import TabularMDPEnvironment


class TabularSamplerABC(ABC):

  @abstractmethod
  def sample(self, episode_length: int, num_states: int) -> TabularMDPEnvironment:
    """
    Creates and returns the benchmark river swim MDP.

    Args:
      episode_length (int): Length of the episode in this environment.
      num_states (int): Number of states in the environment.

    Returns:
      TabularMDPEnvironment
    """
    raise NotImplementedError
