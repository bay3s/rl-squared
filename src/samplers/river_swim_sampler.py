import random
import numpy as np

from .tabular_sampler_abc import TabularSamplerABC
from src.environments import TabularMDPEnvironment


class RiverSwimSampler(TabularSamplerABC):

  def sample(self, episode_length: int, num_states: int) -> TabularMDPEnvironment:
    """
    Creates and returns the benchmark river swim MDP.

    Args:
      episode_length (int): Length of the episode in this environment.
      num_states (int): Number of states in the environment.

    Returns:
      TabularMDPEnvironment
    """
    num_actions = 2
    r_true = dict()
    p_true = dict()

    for s in range(num_states):
      for a in range(num_actions):
        r_true[s, a] = (0, 0)
        p_true[s, a] = np.zeros(num_states)

    # reward probabilities
    r = random.uniform(1, 9)
    r_true[0, 0] = (r / 1000, 0)
    r_true[num_states - 1, 1] = (1, 0)

    # transitions
    for s in range(num_states):
      p_true[s, 0][max(0, s - 1)] = 1.

    t1 = random.uniform(0, 1)
    t2 = random.uniform(0, 1 - t1)

    for s in range(1, num_states - 1):
      p_true[s, 1][min(num_states - 1, s + 1)] = t1
      p_true[s, 1][s] = t2
      p_true[s, 1][max(0, s - 1)] = 1 - t1 - t2

    t3 = random.uniform(0, 1)
    p_true[0, 1][0] = t3
    p_true[0, 1][1] = 1 - t3
    p_true[num_states - 1, 1][num_states - 1] = 1 - t3
    p_true[num_states - 1, 1][num_states - 2] = t3

    river_swim = TabularMDPEnvironment(num_states, num_actions, episode_length)
    river_swim.R = r_true
    river_swim.P = p_true
    river_swim.reset()

    return river_swim

