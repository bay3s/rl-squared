"""
Implementation of a Tabular MDP Environment.

R - dict by (s,a) - each R[s,a] = (mean_rewards, sd_reward)
P - dict by (s,a) - each P[s,a] = transition vector size S

References:
  - https://github.com/iosband/TabulaRL/blob/master/src/environment.py

Original Author:
  - iosband@stanford.edu
"""
