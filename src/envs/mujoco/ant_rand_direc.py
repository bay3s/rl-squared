from typing import Tuple, Any

import numpy as np
from gym.envs.mujoco import AntEnv
from gym.utils.ezpickle import EzPickle

from src.envs.base import MetaEnv


class AntRandDirecEnv(MetaEnv, AntEnv, EzPickle):

  def __init__(self, goal_direction: float = None):
    """
    Initialize the environment.

    Args:
      goal_direction (float): Direction of the goal.
    """
    MetaEnv.__init__(self)
    AntEnv.__init__(self)
    EzPickle.__init__(self)

    self.goal_direction = goal_direction if goal_direction else 1.0
    self.render_mode = 'rgb_array'
    pass

  def sample_tasks(self, num_tasks: int) -> np.array:
    """
    Sample tasks from the meta environment.

    Args:
        num_tasks (int): Number of tasks to sample for the given environment.

    Returns:
      np.array
    """
    return np.random.choice((-1.0, 1.0), (num_tasks,))

  def set_task(self, task: Any) -> None:
    """
    Args:
      task: task of the meta-learning environment
    """
    self.goal_direction = task

  def get_task(self):
    """
    Returns:
      task: task of the meta-learning environment
    """
    return self.goal_direction

  def step(self, a) -> Tuple:
    """
    Take a step in the environment.

    Args:
      a (float): Direction in which to step.

    Returns:
      Tuple
    """
    xposbefore = self.get_body_com("torso")[0]
    self.do_simulation(a, self.frame_skip)

    xposafter = self.get_body_com("torso")[0]
    forward_reward = self.goal_direction * (xposafter - xposbefore) / self.dt
    ctrl_cost = .5 * np.square(a).sum()
    contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

    survive_reward = 1.0
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    state = self.state_vector()
    notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.
    done = not notdone
    ob = self._get_obs()

    return ob, reward, done, dict(
      reward_forward = forward_reward,
      reward_ctrl = -ctrl_cost,
      reward_contact = -contact_cost,
      reward_survive = survive_reward
    )

  def _get_obs(self):
    return np.concatenate([
      self.sim.data.qpos.flat[2:],
      self.sim.data.qvel.flat,
      np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
    ])

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(size = self.model.nq, low = -.1, high = .1)
    qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    self.set_state(qpos, qvel)

    return self._get_obs()

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5


if __name__ == '__main__':
  env = AntRandDirecEnv()
  env.render_mode = 'human'

  while True:
    task = env.sample_tasks(1)[0]
    env.set_task(task)
    env.reset()

    for _ in range(100):
      env.render()
      _, reward, _, _ = env.step(env.action_space.sample())
      pass
