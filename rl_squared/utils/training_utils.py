import os
from typing import List, Tuple
from datetime import datetime
import pathlib

import numpy as np
import torch
import torch.nn as nn

from rl_squared.envs.pytorch_vec_env_wrapper import PyTorchVecEnvWrapper
from rl_squared.networks.stateful.stateful_actor_critic import StatefulActorCritic
from rl_squared.training.meta_episode_batch import MetaEpisodeBatch


@torch.no_grad()
def sample_meta_episodes(
    actor_critic: StatefulActorCritic,
    rl_squared_envs: PyTorchVecEnvWrapper,
    meta_episode_length: int,
    num_meta_episodes: int,
    use_gae: bool,
    gae_lambda: float,
    discount_gamma: float,
    device: torch.device,
) -> Tuple[List[MetaEpisodeBatch], List]:
    """
    Sample meta-episodes in parallel.

    Returns a list of meta-episodes and the mean reward per step.

    Args:
        actor_critic (StatefulActorCritic): Actor-critic to be used for sampling.
        rl_squared_envs (PyTorchVecEnvWrapper): Parallel environments for sampling episodes.
        meta_episode_length (int): Meta-episode length, each "meta-episode" has multiple episodes.
        num_meta_episodes (int): Number of meta-episodes to sample.
        use_gae (bool): Whether to use GAE to compute advantages.
        gae_lambda (float): GAE lambda parameter.
        discount_gamma (float): Discount rate.
        device (torch.device): Device on which to transfer the tensors.

    Returns:
        Tuple[List[MetaEpisodeBatch], float]
    """
    observation_space = rl_squared_envs.observation_space
    action_space = rl_squared_envs.action_space

    recurrent_state_size = actor_critic.recurrent_state_size
    num_parallel_envs = rl_squared_envs.num_envs

    meta_episode_batch = list()
    episode_rewards = list()
    total_env_steps = 0

    for _ in range(num_meta_episodes // num_parallel_envs):
        meta_episodes = MetaEpisodeBatch(
            meta_episode_length,
            num_parallel_envs,
            observation_space,
            action_space,
            recurrent_state_size,
        )

        rl_squared_envs.sample_tasks_async()
        initial_observations = rl_squared_envs.reset()
        meta_episodes.obs[0].copy_(initial_observations)

        for step in range(meta_episode_length):
            (
                value_preds,
                actions,
                action_log_probs,
                recurrent_states_actor,
                recurrent_states_critic,
            ) = actor_critic.act(
                meta_episodes.obs[step].to(device),
                meta_episodes.recurrent_states_actor[step].to(device),
                meta_episodes.recurrent_states_critic[step].to(device),
            )

            obs, rewards, dones, infos = rl_squared_envs.step(actions)

            # rewards
            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            done_masks = torch.FloatTensor(
                [[0.0] if _done else [1.0] for _done in dones]
            )

            # insert
            meta_episodes.insert(
                obs,
                recurrent_states_actor,
                recurrent_states_critic,
                actions,
                action_log_probs,
                value_preds,
                rewards,
                done_masks,
            )

            # num steps
            total_env_steps += num_parallel_envs
            pass

        next_value_pred, _ = actor_critic.get_value(
            meta_episodes.obs[-1].to(device),
            meta_episodes.recurrent_states_critic[-1].to(device),
        )

        next_value_pred.detach()

        meta_episodes.compute_returns(
            next_value_pred, use_gae, discount_gamma, gae_lambda
        )

        meta_episode_batch.append(meta_episodes)
        pass

    mean_reward_per_step = np.sum(episode_rewards) / total_env_steps

    return meta_episode_batch, mean_reward_per_step


def save_checkpoint(
    iteration: int,
    checkpoint_dir: str,
    checkpoint_name: str,
    actor: nn.Module,
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Saves a checkpoint of the latest actor, critic, optimizer.

    Args:
        iteration (int): Number of training iterations so far.
        checkpoint_dir (str): Directory for checkpointing.
        checkpoint_name (str): Model name for checkpointing.
        actor (nn.Module): Actor in the actor-critic setup.
        critic (nn.Module): Critic in the actor-critic setup.
        optimizer (torch.optim.Optimizer): Optimizer used.

    Returns:
        None
    """
    if not os.path.exists(checkpoint_dir):
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = f"{checkpoint_dir}/checkpoint-{checkpoint_name}.pt"

    # save
    torch.save(
        {
            "iteration": iteration,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    pass


def timestamp() -> int:
    """
    Return the current timestamp in integer format.

    Returns:
        int
    """
    return int(datetime.timestamp(datetime.now()))
