from typing import Tuple

import torch

import gym


def _flatten_helper(T: int, N: int, _tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten a given tensor containing rollout information.

    Args:
        T (int): Corresponds to the number of steps in the rollout.
        N (int): Number of processes running.
        _tensor (torch.Tensor): Tensor to flatten.

    Returns:
        torch.Tensor
    """
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage:

    def __init__(
        self,
        rollout_steps: int,
        num_processes: int,
        obs_shape: Tuple,
        action_space: gym.Space,
        recurrent_state_size: int,
    ):
        """
        Initialize the rollout storage.

        Args:
            rollout_steps (int): Number of *steps per rollout*.
            num_processes (int): Number of parallel processes.
            obs_shape (Tuple): Observation shape.
            action_space (gym.Space): Action space for the environment.
            recurrent_state_size (int): Recurrent state size for a memory-augmented agent.
        """
        self.rollout_steps = rollout_steps
        self.num_processes = num_processes
        self.recurrent_state_size = recurrent_state_size
        self.obs_shape = obs_shape
        self.action_space = action_space

        # init
        self.step = 0

        # general
        self.obs = None
        self.recurrent_states = None
        self.rewards = None
        self.returns = None
        self.value_preds = None
        self.action_log_probs = None
        self.actions = None

        # masks
        self.done_masks = None
        self.time_limit_masks = None
        self.recurrent_state_masks = None

        # reset
        self.reset()
        pass

    def reset(self) -> None:
        """
        Resets the storage.

        Returns:
            None
        """
        self.step = 0

        # initial state
        self.obs = torch.zeros(self.rollout_steps + 1, self.num_processes, *self.obs_shape)

        # @todo check if this is the right initialization to use.
        self.recurrent_states = torch.zeros(
            self.rollout_steps + 1, self.num_processes, self.recurrent_state_size
        )

        self.recurrent_states[0] = torch.ones(self.recurrent_states[0].shape)

        self.rewards = torch.zeros(self.rollout_steps, self.num_processes, 1)
        self.value_preds = torch.zeros(self.rollout_steps + 1, self.num_processes, 1)
        self.returns = torch.zeros(self.rollout_steps + 1, self.num_processes, 1)
        self.action_log_probs = torch.zeros(self.rollout_steps, self.num_processes, 1)

        if self.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
            self.actions = torch.zeros(self.rollout_steps, self.num_processes, action_shape).long()
        elif self.action_space.__class__.__name__ == "Box":
            action_shape = self.action_space.shape[0]
            self.actions = torch.zeros(self.rollout_steps, self.num_processes, action_shape)
        else:
            raise NotImplementedError

        # masks
        self.done_masks = torch.ones(self.rollout_steps + 1, self.num_processes, 1)
        self.time_limit_masks = torch.ones(self.rollout_steps + 1, self.num_processes, 1)
        self.recurrent_state_masks = torch.ones(self.rollout_steps + 1, self.num_processes, 1)

        # reset state
        # self.recurrent_state_masks[0] = torch.zeros(self.recurrent_state_masks[0].shape)
        pass

    def to(self, device: torch.device) -> None:
        """
        Transfer the tensors to a specific device.

        Args:
            device (torch.device): Torch device on which to transfer the tensors.

        Returns:
            None
        """
        self.obs = self.obs.to(device)
        self.recurrent_states = self.recurrent_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)

        self.done_masks = self.done_masks.to(device)
        self.time_limit_masks = self.time_limit_masks.to(device)
        self.recurrent_state_masks = self.recurrent_state_masks.to(device)
        pass

    def insert(
        self,
        obs: torch.Tensor,
        recurrent_hidden_states: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        done_masks: torch.Tensor,
        time_limit_masks: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
    ):
        """
        Insert transition details into storage.

        Args:
            obs (torch.Tensor): Observations to be inserted.
            recurrent_hidden_states (torch.Tensor): Recurrent hidden states to be inserted.
            actions (torch.Tensor): Actions to be inserted.
            action_log_probs (torch.Tensor): Log probabilities of actions to be inserted.
            value_preds (torch.Tensor): Value predictions to be inserted.
            rewards (torch.Tensor): Rewards to be inserted.
            done_masks (torch.Tensor): Done masks to be inserted (0 if done, 1 if not).
            time_limit_masks (torch.Tensor): Time limit recurrent_state_masks to be inserted (0 if time-limit is hit,
            1 if not).
            recurrent_state_masks (torch.Tensor): Recurrent state masks to be inserted (0 to reset the state, 1 if not).

        Returns:
            None
        """
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)

        self.step = (self.step + 1) % self.rollout_steps

        self.done_masks[self.step + 1].copy_(done_masks)
        self.time_limit_masks[self.step + 1].copy_(time_limit_masks)
        self.recurrent_state_masks[self.step + 1].copy_(recurrent_state_masks)
        pass

    def compute_returns(
        self,
        next_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        gae_lambda: float,
        use_proper_time_limits: bool = True,
    ) -> None:
        """
        Compute returns for each of the rollouts.

        Args:
            next_value (torch.Tensor): Next predicted value.
            use_gae (bool): Whether to use GAE for advantage estimates.
            gamma (float): Discount gamme to be used.
            gae_lambda (float): GAE lambda value.
            use_proper_time_limits (bool): Whether to use proper time limits for end of episode.

        Returns:
            None
        """
        if use_proper_time_limits:
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
                    gae = gae * self.time_limit_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.done_masks[step + 1]
                        + self.rewards[step]
                    ) * self.time_limit_masks[step + 1] + (
                        1 - self.time_limit_masks[step + 1]
                    ) * self.value_preds[
                        step
                    ]
        else:
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

    def minibatches(self, advantages: torch.Tensor, num_minibatches: int):
        """
        Multiprocessing compatible mini-batch generator for training the policy of an agent.

        Args:
            advantages (torch.Tensor): Computed advatages.
            num_minibatches (int): Number of minibatches.

        Yields:
            Tuple
        """
        num_processes = self.rewards.size(1)
        assert num_processes >= num_minibatches, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_minibatches)
        )

        num_envs_per_batch = num_processes // num_minibatches
        perm = torch.randperm(num_processes)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            done_masks_batch = []
            recurrent_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                done_masks_batch.append(self.done_masks[:-1, ind])
                recurrent_masks_batch.append(self.recurrent_state_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

                # @todo returns the first hidden state.
                recurrent_states_batch.append(self.recurrent_states[0:1, ind])
                pass

            T, N = self.rollout_steps, num_envs_per_batch

            # these are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            done_masks_batch = torch.stack(done_masks_batch, 1)
            recurrent_masks_batch = torch.stack(recurrent_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # states is just a (N, -1) tensor
            recurrent_states_batch = torch.stack(
                recurrent_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            done_masks_batch = _flatten_helper(T, N, done_masks_batch)
            recurrent_masks_batch = _flatten_helper(T, N, recurrent_masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )

            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_states_batch, actions_batch, value_preds_batch, return_batch, \
                done_masks_batch, recurrent_masks_batch, old_action_log_probs_batch, adv_targ
