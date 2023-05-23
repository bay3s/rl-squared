from collections import OrderedDict
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Sequence, Type, Union, Tuple

import numpy as np

import gym
import gym.spaces as spaces


from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


def _flatten_obs(
    obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space
) -> VecEnvObs:
    """
    Worker to use with the environment.

    Args:
        obs (np.array): A list or tuple of observations, one per environment.
        space (spaces.Space): Observation space.

    Returns:
        VecEnvObs
    """
    assert isinstance(
        obs, (list, tuple)
    ), "Expected list or tuple of observations per environment"
    assert len(obs) > 0, "Need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(
            space.spaces, OrderedDict
        ), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict
        ), "Non-dict observation for environment with Dict observation space"
        return OrderedDict(
            [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
        )
    elif isinstance(space, spaces.Tuple):
        assert isinstance(
            obs[0], tuple
        ), "Non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
    else:
        return np.stack(obs)


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    """
    Worker to use with the environment.

    Args:
        remote (mp.connection.Connection): Remote connection.
        parent_remote (mp.connection.Connection): Parent connection.
        env_fn_wrapper (CloudpickleWrapper): Cloudpickle wrapper for the environment instance.

    Returns:
        None
    """
    # import here to avoid circular imports
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.var()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == "sample_task":
                env.sample_task()
            elif cmd == "step":
                observation, reward, done, info = env.step(data)

                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation = env.reset()

                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                if type(observation) is tuple:
                    remote.send(observation[0])
                else:
                    remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class MultiprocessingVecEnv(VecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        """
        Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
        process, allowing significant speed up when the environment is computationally complex.

        Args:
            env_fns (List[Callable]): Functions to create environments that will be run in subprocesses.
        """
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = mp.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()
            pass

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        # VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.num_envs = len(env_fns)
        self.observation_space = observation_space
        self.action_space = action_space
        pass

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def sample_tasks_async(self) -> None:
        """
        Sample a task from the environment.

        Returns:
            None
        """
        for remote in self.remotes:
            remote.send(("sample_task", {}))

        self.waiting = True

    def step_async(self, actions: np.ndarray) -> None:
        """
        Sends a `step` request to environment workers.

        Args:
            actions (np.ndarray): Numpy array of actions to take in each of the environments.

        Returns:
            None
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for results of taking a step in each of the environments.

        Returns:
            VecEnvStepReturn
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        return (
            _flatten_obs(obs, self.observation_space),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Set random seed.

        Params:
            seed (int): Random seed to be set.

        Returns:
            None
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))

        return [remote.recv() for remote in self.remotes]

    def reset(self, seed=None) -> VecEnvObs:
        """
        Reset the environment.

        Returns:
            VecEnvObs
        """
        for remote in self.remotes:
            remote.send(("reset", None))

        obs = [remote.recv() for remote in self.remotes]

        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        """
        Close the environment.

        Returns:
          None
        """
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(("close", None))

        for process in self.processes:
            process.join()

        self.closed = True
        pass

    def get_images(self) -> Sequence[np.ndarray]:
        """
        Get images for each of the environments.

        Returns:
            Sequence[np.ndarray]
        """
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))

        imgs = [pipe.recv() for pipe in self.remotes]

        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """
        Return attribute from vectorized environment (see base class).

        Args:
            attr_name (str): Attribute name.
            indices (VecEnvIndices): Indices for the vectorized environments.

        Returns:
            List[Any]
        """
        target_remotes = self._get_target_remotes(indices)

        for remote in target_remotes:
            remote.send(("get_attr", attr_name))

        return [remote.recv() for remote in target_remotes]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """
        Set attribute inside vectorized environments (see base class).

        Args:
            attr_name (str): Name of the attribute to be set.
            value (Any): Value ofthe attribute to be set.
            indices (VecEnvIndices): Indices of the environments for which to set the attribute.

        Returns:
            None
        """
        target_remotes = self._get_target_remotes(indices)

        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))

        for remote in target_remotes:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """
        Call instance methods of vectorized environments.

        Args:
            method_name (str): Name of the method to be called.
            *method_args (Tuple): Arguments for the methods.
            indices (VecEnvIndices): Indices for the environments.
            **method_kwargs (dict): Kwargs for the methods.

        Returns:
            List[Any]
        """
        target_remotes = self._get_target_remotes(indices)

        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))

        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """
        Check if worker environments are wrapped with a given wrapper

        Args:
            wrapper_class (Type[gym.Wrapper]): Wrapper class for the environment.
            indices (VecEnvIndices): Indices for the VecEnvs

        Returns:
            List[bool]
        """
        target_remotes = self._get_target_remotes(indices)

        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))

        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(
        self, indices: VecEnvIndices
    ) -> List[mp.connection.Connection]:
        """
        Get the connection object needed to communicate with the wanted envs that are in subprocesses.

        Args:
            indices (VecEnvIndices): refers to indices of envs.

        Returns:
            List[mp.connection.Connection]
        """
        indices = self._get_indices(indices)

        return [self.remotes[i] for i in indices]
