import os

import torch
import numpy as np
import wandb

import core.utils.logging_utils as logging_utils
from core.training.experiment_config import ExperimentConfig
from core.learners import PPO

from core.utils.env_utils import make_vec_envs
from core.utils.training_utils import sample_meta_episodes
from core.training.meta_batch_sampler import MetaBatchSampler

from core.networks.stateful.stateful_actor_critic import StatefulActorCritic


class Trainer:
    def __init__(self, experiment_config: ExperimentConfig):
        """
        Initialize an instance of a trainer for PPO.

        Args:
            experiment_config (ExperimentConfig): Params to be used for the trainer.
        """
        self.config = experiment_config

        # private
        self._device = None
        self._log_dir = None
        self._eval_log_dir = None
        pass

    def train(self, enable_wandb: bool = True) -> None:
        """
        Train an agent based on the configs specified by the training parameters.

        Args:
            enable_wandb (bool): Whether to log to Wandb, `True` by default.

        Returns:

        """
        # log
        self.save_params()

        if enable_wandb:
            wandb.login()
            wandb.init(project="rl-squared", config=self.config.dict)

        # seed
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)

        # clean
        logging_utils.cleanup_log_dir(self.log_dir)
        logging_utils.cleanup_log_dir(self.eval_log_dir)

        torch.set_num_threads(1)

        rl_squared_envs = make_vec_envs(
            self.config.env_name,
            self.config.env_configs,
            self.config.random_seed,
            self.config.num_processes,
            self.config.discount_gamma,
            self.config.log_dir,
            self.device,
            allow_early_resets=True,
        )

        actor_critic = StatefulActorCritic(
            rl_squared_envs.observation_space,
            rl_squared_envs.action_space,
            recurrent_state_size=256,
        ).to_device(self.device)

        ppo = PPO(
            actor_critic=actor_critic,
            clip_param=self.config.ppo_clip_param,
            opt_epochs=self.config.ppo_opt_epochs,
            num_minibatches=self.config.ppo_num_minibatches,
            value_loss_coef=self.config.ppo_value_loss_coef,
            entropy_coef=self.config.ppo_entropy_coef,
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            eps=self.config.optimizer_eps,
            max_grad_norm=self.config.max_grad_norm,
        )

        for j in range(self.config.policy_iterations):
            # anneal
            if self.config.use_linear_lr_decay:
                ppo.anneal_learning_rates(j, self.config.policy_iterations)
                pass

            # sample
            meta_episode_batches, meta_episode_rewards = sample_meta_episodes(
                actor_critic,
                rl_squared_envs,
                self.config.meta_episode_length,
                self.config.meta_episodes_per_epoch,
                self.config.use_gae,
                self.config.gae_lambda,
                self.config.discount_gamma,
                self.config.use_proper_time_limits,
            )

            minibatch_sampler = MetaBatchSampler(meta_episode_batches)
            value_loss, action_loss, dist_entropy = ppo.update(minibatch_sampler)

            if enable_wandb:
                wandb.log(
                    {
                        "mean_value_loss": value_loss,
                        "mean_action_loss": action_loss,
                        "mean_dist_entropy": dist_entropy,
                        "mean_rewards": np.mean(meta_episode_rewards),
                    }
                )
            pass

        if enable_wandb:
            # end
            wandb.finish()
        pass

    @property
    def log_dir(self) -> str:
        """
        Returns the path for training logs.

        Returns:
            str
        """
        if not self._log_dir:
            self._log_dir = os.path.expanduser(self.config.log_dir)

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
        Save experiment_config to the logging directory.

        Returns:
          None
        """
        self.config.save()
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

        use_cuda = self.config.use_cuda and torch.cuda.is_available()
        if use_cuda and self.config.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        return self._device
