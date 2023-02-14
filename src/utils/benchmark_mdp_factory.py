import numpy as np

from src.environments.tabular_mdp_environment import TabularMDPEnvironment


def benchmark_tabular_mdp(num_states: int = 10, mean_reward: float = 1.0, episode_length: int = 20
                          ) -> TabularMDPEnvironment:
  """
  Generates and returns a regular Tabular MDP environment with rewards following a Gaussian distribution
  with a standard deviation 1 and expected reward = meanReward

  Args:
    num_states (int): Number of states in the MDP.
    mean_reward (float): Mean reward.
    episode_length (int): Length of the episode.

  Returns:
    TabularMDPEnvironment
  """
  num_actions = 2
  R_true = {}
  P_true = {}

  for s in range(num_states):
    for a in range(num_actions):
      R_true[s, a] = (mean_reward, 1)
      P_true[s, a] = np.zeros(num_states)

  # Transitions
  for s in range(num_states):
    P_true[s, 0][max(0, s - 1)] = 1.

  for s in range(1, num_states - 1):
    P_true[s, 1][min(num_states - 1, s + 1)] = 0.35
    P_true[s, 1][s] = 0.6
    P_true[s, 1][max(0, s - 1)] = 0.05

  P_true[0, 1][0] = 0.4
  P_true[0, 1][1] = 0.6
  P_true[num_states - 1, 1][num_states - 1] = 0.6
  P_true[num_states - 1, 1][num_states - 2] = 0.4

  tabular_env = TabularMDPEnvironment(num_states, num_actions, episode_length)
  tabular_env.R = R_true
  tabular_env.P = P_true
  tabular_env.reset()

  return tabular_env

