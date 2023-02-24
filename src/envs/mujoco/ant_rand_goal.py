import numpy as np

from src.utils import logger
from gym.envs.mujoco import AntEnv
from gym.utils.ezpickle import EzPickle

from src.envs.base import MetaEnv


class AntRandGoalEnv(MetaEnv, AntEnv, EzPickle):

  def __init__(self, goal_pos: float = None):
    """
    Initialize the environment.

    Args:
      goal_pos (float): The goal position.
    """
    MetaEnv.__init__(self)
    AntEnv.__init__(self)
    EzPickle.__init__(self)

    self.goal_pos = goal_pos if goal_pos is not None else self.sample_tasks(1)[0]
    pass

  def sample_tasks(self, n_tasks):
    a = np.random.random(n_tasks) * 2 * np.pi
    r = 3 * np.random.random(n_tasks) ** 0.5

    return np.stack((r * np.cos(a), r * np.sin(a)), axis = -1)

  def set_task(self, task):
    """
    Args:
        task: task of the meta-learning environment
    """
    self.goal_pos = task

  def get_task(self):
    """
    Returns:
        task: task of the meta-learning environment
    """
    return self.goal_pos

  def step(self, a):
    self.do_simulation(a, self.frame_skip)
    xposafter = self.get_body_com("torso")
    goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal
    ctrl_cost = .1 * np.square(a).sum()
    contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
    # survive_reward = 1.0
    survive_reward = 0.0
    reward = goal_reward - ctrl_cost - contact_cost + survive_reward
    state = self.state_vector()
    # notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.
    # done = not notdone
    done = False
    ob = self._get_obs()
    return ob, reward, done, dict(
      reward_forward = goal_reward,
      reward_ctrl = -ctrl_cost,
      reward_contact = -contact_cost,
      reward_survive = survive_reward)

  def _get_obs(self):
    return np.concatenate([
      self.sim.data.qpos.flat,
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

  def log_diagnostics(self, paths, prefix = ''):
    progs = [np.mean(path["env_infos"]["reward_forward"]) for path in paths]
    ctrl_cost = [-np.mean(path["env_infos"]["reward_ctrl"]) for path in paths]

    logger.logkv(prefix + 'AverageForwardReturn', np.mean(progs))
    logger.logkv(prefix + 'MaxForwardReturn', np.max(progs))
    logger.logkv(prefix + 'MinForwardReturn', np.min(progs))
    logger.logkv(prefix + 'StdForwardReturn', np.std(progs))

    logger.logkv(prefix + 'AverageCtrlCost', np.mean(ctrl_cost))
