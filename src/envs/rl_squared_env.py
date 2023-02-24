import numpy as np

from src.utils.serializable import Serializable


class RLSquaredEnv(Serializable):

  def __init__(self, env):
    """
    Initializes an environment for RL-Squared.
    """
    Serializable.__init__(self)
    Serializable.quick_init(self, locals())
    self._wrapped_env = env
    pass

  def __getattr__(self, attr):
    """
    If normalized env does not have the attribute then call the attribute in the wrapped_env

    Args:
      attr: attribute to get
    Returns:
      attribute of the wrapped_env
    """
    orig_attr = self._wrapped_env.__getattribute__(attr)

    if callable(orig_attr):
      def hooked(*args, **kwargs):
        result = orig_attr(*args, **kwargs)
        return result

      return hooked
    else:
      return orig_attr

  def reset(self):
    """
    Reset the environment.

    Returns:
      np.ndarray
    """
    obs = self._wrapped_env.reset()

    return np.concatenate([obs, np.zeros(self._wrapped_env.action_space.shape), [0], [0]])

  def __getstate__(self):
    d = Serializable.__getstate__(self)

    return d

  def __setstate__(self, d):
    Serializable.__setstate__(self, d)

  def step(self, action):
    wrapped_step = self._wrapped_env.step(action)
    next_obs, reward, done, info = wrapped_step
    next_obs = np.concatenate([next_obs, action, [reward], [done]])

    return next_obs, reward, done, info
