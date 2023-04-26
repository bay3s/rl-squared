from typing import Tuple, Any
import gym


class TimeLimitEnvWrapper(gym.Wrapper):
    """
    Environment wrapper that modifies the information returned while taking a step to reflect whether the episode was
    done as a result of hitting time limits.
    """

    def step(self, action: Any) -> Tuple:
        """
        Take step in the current environment given the action.

        Adds an extra variable to the info normally returned to indicate whether an episode ended as a result of
        hitting the time limit.

        Args:
            action (Any): Action to be taken in the environment.

        Returns:
            Tuple
        """
        obs, rew, done, info = self.env.step(action)

        # `_max_episode_steps` and `_elapsed_steps` inherited from `gym.wrappers.time_limit.TimeLimit`
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["time_limit_exceeded"] = True

        return obs, rew, done, info

    def reset(self, **kwargs) -> Tuple:
        """
        Reset the environment.

        Args:
            **kwargs (dict): Key word arguments.

        Returns:
            Tuple
        """
        return self.env.reset(**kwargs)
