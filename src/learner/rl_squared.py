import os
import torch.nn as nn
from .rl_squared_configs import RLSquaredConfigs


class RLSquared:

  def __init__(self, actor: nn.Module, critic: nn.Module, env_sampler: callable, device: str, directory: str):
    """
    Implements the core RL-squared algorithm.

    Args:
      actor (nn.Module): Neural net used for selecting actions.
      critic (nn.Module): Neural net used to estimate state values.
      env_sampler (callable): Function used to sample environments from a given distribution for meta-learning.
      device (str): Device to use for training the PyTorch modules.
      directory (str): Directory used for logging and saving models.
    """
    self.actor = actor
    self.critic = critic
    self.sample_environment = env_sampler

    self.device = device
    self.base_directory = directory
    self.logging_directory = f'{self.base_directory}/logging/'
    self.models_directory = f'{self.base_directory}/models/'
    pass

  def meta_train(self, configs: RLSquaredConfigs, plot: bool = True, verbose: bool = True) -> nn.Module:
    """
    Meta-training logic for RL-Squared, returns the fine-tuned agent.

    Args:
      configs (RLSquaredConfigs): Configs for the current meta-training run.
      plot (bool): Whether to plot the progress through training iterations.
      verbose (bool): Whether to output logs to the console / terminal

    Returns:
      nn.Module
    """
    env = self.sample_environment(nState=10, epLen=20)

    agent = PPO(env, policy=self.policy, value=self.value)

    start_time = datetime.datetime.now()

    for t in range(trials):
      env = self.make_env(nState=10, epLen=20)
      print(f"\nTRIAL NUMBER : {t+1}\n")
           # f"\nREWARDS : {env.R} |"
           # f"\nTRANSITION PROBABILITIES : {env.P}")

      agent.env = env
      if reset_hidden_state and self.policy_type == 'recurrent':
        agent.policy.reset_hidden_state()

      log_dir = os.path.join(self.log_dir, f'meta_train/Trial_{t+1}/')
      _ = agent.train(
        epochs=epochs_per_trial,
        episodes_per_epoch=episodes_per_epoch,
        n_value_updates=n_value_updates,
        n_policy_updates=n_policy_updates,
        value_lr=value_lr,
        policy_lr=policy_lr,
        gamma=gamma,
        epsilon=epsilon,
        max_traj_length=max_traj_length,
        log_dir=log_dir,
        RENDER=False,
        PLOT_REWARDS=PLOT_REWARDS,
        VERBOSE=VERBOSE,
      )

  def meta_test(self,
                epochs=MTE_EPOCHS,
                episodes_per_epoch=MTE_EPISODES_PER_EPOCH,
                n_value_updates=MTE_N_VALUE_UPDATES,
                n_policy_updates=MTE_N_POLICY_UPDATES,
                value_lr=MTE_VALUE_FN_LEARNING_RATE,
                policy_lr=MTE_POLICY_LEARNING_RATE,
                gamma=MTE_GAMMA,
                epsilon=MTE_EPSILON,
                max_traj_length=MTE_MAX_TRAJ_LENGTH,
                PLOT_REWARDS=False,
                VERBOSE=True,
                ):
    env = self.make_env(nState=10, epLen=20)
    if self.policy_type == 'recurrent':
      agent = PPOGRU(env, policy=self.policy, value=self.value)
      load_path = os.path.join(self.log_dir, f'meta_train/Trial_{MTR_TRIALS}/PPOGRU_TabularEnv.pt')
    else:
      agent = PPOMLP(env, policy=self.policy, value=self.value)
      load_path = os.path.join(self.log_dir, f'meta_train/Trial_{MTR_TRIALS}/PPOMLP_TabularEnv.pt')
    agent.load(path=load_path)
    save_path = os.path.join(self.log_dir, 'meta_test')

    rewards = agent.train(
      epochs=epochs,
      episodes_per_epoch=episodes_per_epoch,
      n_value_updates=n_value_updates,
      n_policy_updates=n_policy_updates,
      value_lr=value_lr,
      policy_lr=policy_lr,
      gamma=gamma,
      epsilon=epsilon,
      max_traj_length=max_traj_length,
      log_dir=save_path,
      RENDER=False,
      PLOT_REWARDS=PLOT_REWARDS,
      VERBOSE=VERBOSE,
    )

    return rewards
