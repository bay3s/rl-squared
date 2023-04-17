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
from core.utils.training_utils import sample_meta_episodes
from core.training.meta_batch_sampler import MetaBatchSampler

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

    def meta_train(self) -> None:
        """
        Train an agent based on the configs specified by the training parameters.

        Returns:
          None
        """
        # start time
        start = time.time()

        # save
        self.save_params()

        # seed
        torch.manual_seed(self.params.random_seed)
        torch.cuda.manual_seed_all(self.params.random_seed)

        # clean
        logging_utils.cleanup_log_dir(self.log_dir)
        logging_utils.cleanup_log_dir(self.eval_log_dir)

        torch.set_num_threads(1)

        meta_envs = make_vec_envs(
            self.params.env_name,
            self.params.env_configs,
            self.params.random_seed,
            self.params.num_processes,
            self.params.discount_gamma,
            self.params.log_dir,
            self.device,
            allow_early_resets = True,
        )

        actor_critic = StatefulActorCritic(
            meta_envs.observation_space,
            meta_envs.action_space
        ).to_device(self.device)

        ppo = PPO(
            actor_critic=actor_critic,
            clip_param=self.params.ppo_clip_param,
            opt_epochs=self.params.ppo_opt_epochs,
            num_minibatches=self.params.ppo_num_minibatches,
            value_loss_coef=self.params.ppo_value_loss_coef,
            entropy_coef=self.params.ppo_entropy_coef,
            actor_lr=self.params.actor_lr,
            critic_lr=self.params.critic_lr,
            eps=self.params.optimizer_eps,
            max_grad_norm=self.params.max_grad_norm,
        )

        episode_rewards = deque(maxlen=1_000)
        steps_per_epoch = self.params.meta_episode_length * self.params.meta_episodes_per_epoch
        training_epochs = self.params.total_steps // steps_per_epoch

        for j in range(training_epochs):
            # generate episodes
            with torch.no_grad():
                meta_episode_batches = sample_meta_episodes(
                    actor_critic,
                    meta_envs,
                    self.params.meta_episode_length,
                    self.params.meta_episodes_per_epoch,
                    self.params.use_gae,
                    self.params.gae_lambda,
                    self.params.discount_gamma,
                    self.params.use_proper_time_limits
                )

            minibatch_sampler = MetaBatchSampler(meta_episode_batches)
            value_loss, action_loss, dist_entropy = ppo.update(minibatch_sampler)

            # @todo checkpoint
            # @todo meta-evaluate

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
