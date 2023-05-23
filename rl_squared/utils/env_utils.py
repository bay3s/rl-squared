from typing import Callable

import torch
import gym

from gym.envs.registration import register

from rl_squared.envs.multiprocessing_vec_env import MultiprocessingVecEnv
from rl_squared.envs.pytorch_vec_env_wrapper import PyTorchVecEnvWrapper
from rl_squared.envs.rl_squared_env import RLSquaredEnv


def get_render_func(venv: gym.Env):
    """
    Get render function for the environment.

    Args:
        venv (object): Environment in which

    Returns:
        Callable
    """
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def make_env_thunk(
    env_name: str,
    env_configs: dict,
    seed: int,
    rank: int,
) -> Callable:
    """
    Returns a callable to create environments based on the specs provided.

    Args:
        env_name (str): Environment to create.
        env_configs (dict): Key word arguments for making the environment.
        seed (int): Random seed for the experiments.
        rank (int): "Rank" of the environment that the callable would return.

    Returns:
        Callable
    """

    def _thunk():
        env = gym.make(env_name, **env_configs)

        if not callable(getattr(env, "seed", None)):
            raise NotImplementedError(
                f"`seed` required for experiment replicability, but not implemented."
            )

        env.seed(seed + rank)

        if len(env.observation_space.shape) != 1:
            raise NotImplementedError

        env = RLSquaredEnv(env)

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            raise NotImplementedError

        return env

    return _thunk


def make_vec_envs(
    env_name: str,
    env_kwargs: dict,
    seed: int,
    num_processes: int,
    device: torch.device,
) -> PyTorchVecEnvWrapper:
    """
    Returns PyTorch compatible vectorized environments.

    Args:
        env_name (str): Name of the environment to be created.
        env_kwargs (dict): Key word arguments to create the environment.
        seed (int): Random seed for environments.
        num_processes (int): Number of parallel processes to be used for simulations.
        device (torch.device): Device to use with PyTorch tensors.

    Returns:
        PyTorchVecEnvWrapper
    """
    envs = [
        make_env_thunk(env_name, env_kwargs, seed, rank)
        for rank in range(num_processes)
    ]

    envs = MultiprocessingVecEnv(envs)
    envs = PyTorchVecEnvWrapper(envs, device)

    return envs


def register_custom_envs() -> None:
    """
    Register custom environments for experiments.

    Returns:
        None
    """
    register(
        id="BernoulliBandit-v1",
        entry_point="rl_squared.envs.bandits.bernoulli_bandit_env:BernoulliBanditEnv",
    )

    register(
        id="TabularMDP-v1", entry_point="rl_squared.envs.mdps.tabular_env:TabularMDPEnv"
    )

    register(
        id="PointRobotNavigation-v1",
        entry_point="rl_squared.envs.point_robot.navigation_env:NavigationEnv",
    )

    register(
        id="AntTargetPosition-v1",
        entry_point="rl_squared.envs.ant.ant_target_position_env:AntTargetPositionEnv",
    )

    register(
        id="AntTargetVelocity-v1",
        entry_point="rl_squared.envs.ant.ant_target_velocity_env:AntTargetVelocityEnv",
    )

    register(
        id="CheetahTargetVelocity-v1",
        entry_point="rl_squared.envs.cheetah.cheetah_target_velocity_env:CheetahTargetVelocityEnv",
    )
