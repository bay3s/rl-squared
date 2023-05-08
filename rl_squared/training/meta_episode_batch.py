import torch
import gym


class MetaEpisodeBatch:
    def __init__(
        self,
        meta_episode_length: int,
        num_meta_episodes: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        recurrent_state_size: int,
    ):
        """
        Initialize the rollout storage.

        Args:
            meta_episode_length (int): Number of steps per meta-episode.
            num_meta_episodes (int): Number of meta episodes.
            observation_space (int): Dimensions of the observation space.
            action_space (gym.Space): Action space for the environment.
            recurrent_state_size (int): Recurrent state size for a memory-augmented agent.
        """
        self.meta_episode_length = meta_episode_length
        self.step = 0

        self.obs = torch.zeros(
            meta_episode_length + 1, num_meta_episodes, *observation_space.shape
        )
        self.rewards = torch.zeros(meta_episode_length, num_meta_episodes, 1)
        self.value_preds = torch.zeros(meta_episode_length + 1, num_meta_episodes, 1)
        self.returns = torch.zeros(meta_episode_length + 1, num_meta_episodes, 1)
        self.action_log_probs = torch.zeros(meta_episode_length, num_meta_episodes, 1)
        self.actions = self._init_actions(
            action_space, meta_episode_length, num_meta_episodes
        )

        # recurrent states
        self.recurrent_states_actor = torch.zeros(
            meta_episode_length + 1, num_meta_episodes, recurrent_state_size
        )
        self.recurrent_states_critic = torch.zeros(
            meta_episode_length + 1, num_meta_episodes, recurrent_state_size
        )

        # masks
        self.done_masks = torch.ones(meta_episode_length + 1, num_meta_episodes, 1)
        pass

    @staticmethod
    def _init_actions(
        action_space: gym.Space, meta_episode_length: int, num_meta_episodes: int
    ) -> torch.Tensor:
        """
        Init actions based on the action space.

        Args:
            action_space (gym.Space): Action space for the meta-episodes.
            meta_episode_length (int): Meta-episode length.
            num_meta_episodes (int): Number of meta-episodes to sample.

        Returns:
            torch.Tensor
        """
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
            return torch.zeros(meta_episode_length, num_meta_episodes, action_shape)
        elif action_space.__class__.__name__ == "Box":
            action_shape = action_space.shape[0]
            return torch.zeros(meta_episode_length, num_meta_episodes, action_shape)
        else:
            raise NotImplementedError

    def to_device(self, device: torch.device) -> None:
        """
        Transfer the tensors to a specific device.

        Args:
            device (torch.device): Torch device on which to transfer the tensors.

        Returns:
            None
        """
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)

        self.recurrent_states_actor = self.recurrent_states_actor.to(device)
        self.recurrent_states_critic = self.recurrent_states_critic.to(device)

        self.done_masks = self.done_masks.to(device)
        pass

    def insert(
        self,
        obs: torch.Tensor,
        recurrent_states_actor: torch.Tensor,
        recurrent_states_critic: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        done_masks: torch.Tensor,
    ):
        """
        Insert transition details into storage.

        Args:
            obs (torch.Tensor): Observations to be inserted.
            recurrent_states_actor (torch.Tensor): Recurrent hidden states of the actor to be inserted.
            recurrent_states_critic (torch.Tensor): Recurrent hidden states of the critic to be inserted.
            actions (torch.Tensor): Actions to be inserted.
            action_log_probs (torch.Tensor): Log probabilities of actions to be inserted.
            value_preds (torch.Tensor): Value predictions to be inserted.
            rewards (torch.Tensor): Rewards to be inserted.
            done_masks (torch.Tensor): Done masks to be inserted (0 if done, 1 if not).

        Returns:
            None
        """
        if self.step > self.meta_episode_length:
            raise IndexError(f"Number of steps exceeded.")

        # states
        self.recurrent_states_actor[self.step + 1].copy_(recurrent_states_actor)
        self.recurrent_states_critic[self.step + 1].copy_(recurrent_states_critic)

        # obs, r
        self.obs[self.step + 1].copy_(obs)

        # masks
        self.done_masks[self.step + 1].copy_(done_masks)

        # actions, log probs, value preds, rewards to the current one.
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)

        # update
        self.step = self.step + 1
        pass

    def compute_returns(
        self, next_value: torch.Tensor, use_gae: bool, gamma: float, gae_lambda: float
    ) -> None:
        """
        Compute returns for each of the rollouts.

        Args:
            next_value (torch.Tensor): Next predicted value.
            use_gae (bool): Whether to use GAE for advantage estimates.
            gamma (float): Discount gamme to be used.
            gae_lambda (float): GAE lambda value.

        Returns:
            None
        """
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.done_masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * gae_lambda * self.done_masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.done_masks[step + 1]
                    + self.rewards[step]
                )
