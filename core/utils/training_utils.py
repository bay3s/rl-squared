# @todo move
from typing import List

import torch

from core.envs.pytorch_vec_env_wrapper import PyTorchVecEnvWrapper
from core.networks.stateful.stateful_actor_critic import StatefulActorCritic

from core.training.meta_episode_batch import MetaEpisodeBatch


@torch.no_grad()
def sample_meta_episodes(
    actor_critic: StatefulActorCritic,
    rl_squared_envs: PyTorchVecEnvWrapper,
    meta_episode_length: int,
    num_meta_episodes: int,
    use_gae: bool,
    gae_lambda: float,
    discount_gamma: float,
    use_proper_time_limits: bool
) -> List[MetaEpisodeBatch]:
    observation_space = rl_squared_envs.observation_space
    action_space = rl_squared_envs.action_space

    recurrent_state_size = actor_critic.recurrent_state_size
    num_parallel_envs = rl_squared_envs.num_envs

    meta_episodes_all = list()

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
            # @todo deal with masks differently for RL-Squared
            (
                value_preds,
                actions,
                action_log_probs,
                recurrent_states,
            ) = actor_critic.act(
                meta_episodes.obs[step],
                meta_episodes.recurrent_states[step],
                meta_episodes.done_masks[step]
            )

            obs, rewards, dones, infos = rl_squared_envs.step(actions)

            # masks
            time_limit_masks = torch.FloatTensor(
                [
                    [0.] if "time_limit_exceeded" in info.keys() else [1.]
                    for info in infos
                ]
            )

            done_masks = torch.FloatTensor(
                [
                    [0.] if _done else [1.] for _done in dones
                ]
            )

            # insert
            meta_episodes.insert(
                obs,
                recurrent_states,
                actions,
                action_log_probs,
                value_preds,
                rewards,
                done_masks,
                time_limit_masks,
            )
            pass

        next_value_pred = actor_critic.get_value(
            meta_episodes.obs[-1],
            meta_episodes.recurrent_states[-1],
            meta_episodes.done_masks[-1],
        ).detach()

        meta_episodes.compute_returns(
            next_value_pred,
            use_gae,
            discount_gamma,
            gae_lambda,
            use_proper_time_limits
        )

        meta_episodes_all.append(meta_episodes)
        pass

    return meta_episodes_all
