import os
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd

import core.utils.env_utils as env_utils
import core.utils.logging_utils as logging_utils
from core.training.training_args import TrainingArgs
from core.learners import PPO

from core.utils.env_utils import make_vec_envs
from core.learners.rollout_storage import RolloutStorage

from core.networks.base_actor_critic import BaseActorCritic
from core.networks.stateful.stateful_actor_critic import StatefulActorCritic


class Trainer:

    def __init__(self, params: TrainingArgs):
        """
        Initialize an instance of a trainer for PPO.

        Args:
          params (TrainingArgs): Params to be used for the trainer.
        """
        self.params = params

        # private
        self._device = None
        self._log_dir = None
        self._eval_log_dir = None
        pass

    def train(self) -> None:
        """
        Train an agent based on the configs specified by the training parameters.

        Returns:
          None
        """
        # save
        self.save_params()

        # seed
        torch.manual_seed(self.params.random_seed)
        torch.cuda.manual_seed_all(self.params.random_seed)

        # clean
        logging_utils.cleanup_log_dir(self.log_dir)
        logging_utils.cleanup_log_dir(self.eval_log_dir)

        torch.set_num_threads(1)

        envs = make_vec_envs(
            self.params.env_name,
            self.params.env_configs,
            self.params.random_seed,
            self.params.num_processes,
            self.params.discount_gamma,
            self.params.log_dir,
            self.device,
            False,
        )

        actor_critic = StatefulActorCritic(
            envs.observation_space, envs.action_space
        ).to_device(self.device)

        ppo = PPO(
            actor_critic=actor_critic,
            clip_param=self.params.ppo_clip_param,
            num_epochs=self.params.ppo_num_epochs,
            num_minibatches=self.params.ppo_num_minibatches,
            value_loss_coef=self.params.ppo_value_loss_coef,
            entropy_coef=self.params.ppo_entropy_coef,
            actor_lr=self.params.actor_lr,
            critic_lr=self.params.critic_lr,
            eps=self.params.optimizer_eps,
            max_grad_norm=self.params.max_grad_norm,
        )

        num_steps_per_rollout = self.params.steps_per_trial // self.params.num_processes

        rollouts = RolloutStorage(
            num_steps_per_rollout,
            self.params.num_processes,
            envs.observation_space.shape,
            envs.action_space,
            actor_critic.recurrent_state_size,
        )

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(self.device)

        episode_rewards = deque(maxlen=10)

        start = time.time()

        total_updates = (
            int(self.params.num_env_steps)
            // num_steps_per_rollout
            // self.params.num_processes
        )

        for j in range(total_updates):
            # decay
            if self.params.use_linear_lr_decay:
                ppo.update_linear_schedule(j, total_updates)
                pass

            # @todo sample meta-tasks
            print('Sample new task.')

            # rollouts
            for step in range(num_steps_per_rollout):
                with torch.no_grad():
                    (
                        value,
                        action,
                        action_log_prob,
                        recurrent_hidden_states,
                    ) = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.recurrent_state_masks[step],
                    )

                # step
                obs, reward, done, infos = envs.step(action)

                for info in infos:
                    if "episode" in info.keys():
                        # @todo check `BaseMetaEnv` compatibility, this is set in `Monitor`.
                        episode_rewards.append(info["episode"]["r"])

                # done
                done_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]
                )

                time_limit_masks = torch.FloatTensor(
                    [
                        [0.0] if "time_limit_exceeded" in info.keys() else [1.0]
                        for info in infos
                    ]
                )

                recurrent_state_masks = torch.FloatTensor(
                    [
                        [1.0] for _ in reward
                    ]
                )

                rollouts.insert(
                    obs,
                    recurrent_hidden_states,
                    action,
                    action_log_prob,
                    value,
                    reward,
                    done_masks,
                    time_limit_masks,
                    recurrent_state_masks
                )
                pass

            # value
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.recurrent_state_masks[-1],
                ).detach()

            # returns
            rollouts.compute_returns(
                next_value,
                self.params.use_gae,
                self.params.discount_gamma,
                self.params.gae_lambda,
                self.params.use_proper_time_limits,
            )

            value_loss, action_loss, dist_entropy = ppo.update(rollouts)
            rollouts.reset()

            # checkpoint
            if j % self.params.checkpoint_interval == 0 or j == total_updates - 1:
                try:
                    os.makedirs(self.params.checkpoint_dir)
                except OSError:
                    pass

                torch.save(
                    [
                        actor_critic,
                        getattr(env_utils.get_vec_normalize(envs), "obs_rms", None),
                    ],
                    os.path.join(self.params.checkpoint_dir, "model.pt"),
                )

            # log
            if j % self.params.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (
                    (j + 1) * self.params.num_processes * num_steps_per_rollout
                )
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n "
                    "Last {} training episodes: mean/median reward "
                    "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                        j,
                        total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                        dist_entropy,
                        value_loss,
                        action_loss,
                    )
                )

            # evaluate
            # if (
            #     self.params.eval_interval is not None
            #     and len(episode_rewards) > 1
            #     and j % self.params.eval_interval == 0
            # ):
            #     obs_rms = env_utils.get_vec_normalize(envs).obs_rms
            #     self.evaluate(actor_critic, obs_rms)
            #     pass

    def evaluate(
        self,
        actor_critic: BaseActorCritic,
        obs_rms: RunningMeanStd,
        deterministic: bool = False
    ):
        """
        Evaluate the agent.

        Args:
            actor_critic (BaseActorCritic): Actor-Critic to be tuned using PPO.
            obs_rms (RunningMeanStd): The stablebaselines environment's `running_mean_std`.
            deterministic (bool): Whether to take actions deterministically.

        Returns:
            None
        """
        eval_envs = make_vec_envs(
            self.params.env_name,
            self.params.env_configs,
            self.params.random_seed + self.params.num_processes,
            self.params.num_processes,
            None,
            self.eval_log_dir,
            self.device,
            True,
        )

        vec_norm = env_utils.get_vec_normalize(eval_envs)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms

        eval_episode_rewards = []

        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(
            self.params.num_processes,
            actor_critic.recurrent_state_size,
            device=self.device,
        )
        eval_masks = torch.zeros(self.params.num_processes, 1, device=self.device)

        # @todo add `num_eval_episodes` or `num_eval_steps` to `TrainingArgs`
        while len(eval_episode_rewards) < 1_000:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs, eval_recurrent_hidden_states, eval_masks, deterministic = deterministic
                )

            obs, _, done, infos = eval_envs.step(action)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=self.device,
            )

            for info in infos:
                if "episode" in info.keys():
                    eval_episode_rewards.append(info["episode"]["r"])

        eval_envs.close()

        print(
            "Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(eval_episode_rewards), np.mean(eval_episode_rewards)
            )
        )
        pass

    @property
    def log_dir(self) -> str:
        """
        Returns the path for training logs.

        Returns:
            str
        """
        if not self._log_dir:
            self._log_dir = os.path.expanduser(self.params.log_dir)

        return self._log_dir

    @property
    def eval_log_dir(self):
        """
        Returns the path for evaluation logs.

        Returns:
            str
        """
        if not self._eval_log_dir:
            self._eval_log_dir = self.log_dir + "_eval"

        return self._eval_log_dir

    def save_params(self) -> None:
        """
        Save params to the logging directory.

        Returns:
          None
        """
        self.params.save()
        pass

    @property
    def device(self) -> torch.device:
        """
        Torch device to use for training and optimization.

        Returns:
          torch.device
        """
        if isinstance(self._device, torch.device):
            return self._device

        use_cuda = self.params.use_cuda and torch.cuda.is_available()
        if use_cuda and self.params.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        return self._device
