import numpy as np

from gym.envs.mujoco import AntEnv as AntEnv_


class BaseAntEnv(AntEnv_):
    def __init__(self):
        AntEnv_.__init__(self)
        self._action_scaling = None
        pass

    @property
    def action_scaling(self):
        if (not hasattr(self, "action_space")) or (self.action_space is None):
            return 1.0

        if self._action_scaling is None:
            lb, ub = self.action_space.low, self.action_space.high
            self._action_scaling = 0.5 * (ub - lb)

        return self._action_scaling

    def _get_obs(self):
        return (
            np.concatenate(
                [
                    self.sim.data.qpos.flat,
                    self.sim.data.qvel.flat,
                    np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                    self.sim.data.get_body_xmat("torso").flat,
                    self.get_body_com("torso").flat,
                ]
            )
            .astype(np.float32)
            .flatten()
        )

    def viewer_setup(self):
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True

    def render(self, mode: str = "human"):
        if mode == "rgb_array":
            self._get_viewer(mode).render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return data
        elif mode == "human":
            self._get_viewer(mode).render()
