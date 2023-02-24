from typing import Tuple
import numpy as np

from gym.envs.mujoco import Walker2dEnv
from gym.utils.ezpickle import EzPickle

from src.envs.base import MetaEnv


class Walker2DRandVelEnv(MetaEnv, Walker2dEnv, EzPickle):

  def __init__(self):
    """
    Initialize the Walker environment.
    """
    self.set_task(self.sample_tasks(1)[0])

    MetaEnv.__init__(self)
    Walker2dEnv.__init__(self)
    EzPickle.__init__(self)
    pass

  def sample_tasks(self, n_tasks: int) -> np.ndarray:
    """
    Sample tasks from the environment.

    Args:
      n_tasks (int): Number of tasks to sample from the environment.

    Returns:
      np.ndarray
    """
    return np.random.uniform(0.0, 10.0, (n_tasks,))

  def set_task(self, task) -> None:
    """
    Args:
        task: task of the meta-learning environment
    """
    self.goal_velocity = task

  def get_task(self):
    """
    Returns:
        task: task of the meta-learning environment
    """
    return self.goal_velocity

  def step(self, a) -> Tuple:
    """
    Take one step in the environment.

    Args:
      a (float): Direction in which to step.

    Returns:
      Tuple
    """
    posbefore = self.sim.data.qpos[0]
    self.do_simulation(a, self.frame_skip)
    posafter, height, ang = self.sim.data.qpos[0:3]
    alive_bonus = 15.0
    forward_vel = (posafter - posbefore) / self.dt

    reward = - np.abs(forward_vel - self.goal_velocity)
    reward += alive_bonus
    reward -= 1e-3 * np.square(a).sum()

    done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
    ob = self._get_obs()

    return ob, reward, done, {}

  def _get_obs(self) -> np.ndarray:
    """
    Returns an observation in the environment.

    Returns:
      np.ndarray
    """
    qpos = self.sim.data.qpos
    qvel = self.sim.data.qvel

    return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

  def reset_model(self) -> np.ndarray:
    """
    Reset the environment.

    Returns:
      np.ndarray
    """
    self.set_state(
      self.init_qpos + self.np_random.uniform(low = -.005, high = .005, size = self.model.nq),
      self.init_qvel + self.np_random.uniform(low = -.005, high = .005, size = self.model.nv)
    )

    return self._get_obs()

  def viewer_setup(self) -> None:
    """
    Set up the viewer if the mode is set to human.

    Returns:
      None
    """
    self.viewer.cam.trackbodyid = 2
    self.viewer.cam.distance = self.model.stat.extent * 0.5

