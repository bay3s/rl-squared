"""
Implementation of a Tabular MDP Environment.

R - dict by (s,a) - each R[s,a] = (mean_reward, sd_reward)
P - dict by (s,a) - each P[s,a] = transition vector size S

References:
  - https://github.com/iosband/TabulaRL/blob/master/src/environment.py

Original Author:
  - iosband@stanford.edu
"""
import numpy as np
from numpy import ndarray

from .environment_abc import EnvironmentABC
from typing import Tuple, Any, Union


class TabularMDPEnvironment(EnvironmentABC):

  def __init__(self, num_states: int, num_actions: int, episode_length: int):
    """
    Initializes a tabular episodic MDP.

    Args:
      num_states (int): Number of states.
      num_actions (int): Number of actions.
      episode_length (int): Length of an episode.
    """
    super(TabularMDPEnvironment, self).__init__()

    self.num_states = num_states
    self.num_actions = num_actions
    self.episode_length = episode_length

    self.timestep = 0
    self.state = 0

    # Now initialize R and P
    self.R = {}
    self.P = {}

    for state in range(num_states):
      for action in range(num_actions):
        self.R[state, action] = (1, 1)
        self.P[state, action] = np.ones(num_states) / num_states

  def reset(self) -> int:
    """
    Resets the environment and returns the state.

    Returns:
      int
    """
    self.timestep = 0
    self.state = 0

    return self.state

  def advance(self, action) -> Tuple[Any, Union[int, ndarray, float, complex], bool]:
    """
    Move one step in the environment.

    Args:
      action (int): The action to take given the current state of the environment.

    Returns:
      Tuple[float, int, bool]
    """
    if self.R[self.state, action][1] < 1e-9:
      # hack for no noise
      reward = self.R[self.state, action][0]
    else:
      reward = np.random.normal(loc = self.R[self.state, action][0], scale = self.R[self.state, action][1])

    new_state = np.random.choice(self.num_states, p = self.P[self.state, action])

    # update the environment
    self.state = new_state
    self.timestep += 1

    is_done = self.timestep == self.episode_length

    if is_done:
      self.reset()

    return new_state, reward, is_done

  def compute_q_values(self):
    """
    Compute the Q-values for the environment.

    - q_values - q_values[state, timestep] is vector of Q values for each action.
    - q_max - q_max[timestep] is the vector of optimal values at timestep.

    Returns:
      Tuple[dict, dict]
    """
    q_values = dict()
    q_max = dict()

    q_max[self.episode_length] = np.zeros(self.num_states)

    for i in range(self.episode_length):
      j = self.episode_length - i - 1
      q_max[j] = np.zeros(self.num_states)

      for s in range(self.num_states):
        q_values[s, j] = np.zeros(self.num_actions)

        for a in range(self.num_actions):
          q_values[s, j][a] = self.R[s, a][0] + np.dot(self.P[s, a], q_max[j + 1])

        q_max[j][s] = np.max(q_values[s, j])

    return q_values, q_max
