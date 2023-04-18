import torch
from typing import List


from core.training.meta_episode_batch import MetaEpisodeBatch


class MetaBatchSampler:

  def __init__(self, batches: List[MetaEpisodeBatch]):
    self.meta_episode_batches = batches

    self.obs = self._concat('obs')
    self.rewards = self._concat('rewards')
    self.value_preds = self._concat('value_preds')
    self.returns = self._concat('returns')
    self.action_log_probs = self._concat('action_log_probs')
    self.actions = self._concat('actions')

    self.recurrent_states = self._concat('recurrent_states')
    self.done_masks = self._concat('done_masks')
    self.time_limit_masks = self._concat('time_limit_masks')

    pass

  def _concat(self, attr: str) -> torch.Tensor:
    tensors = [getattr(meta_episode_batch, attr) for meta_episode_batch in self.meta_episode_batches]

    return torch.cat(tensors = tensors, dim = 1)

  def sample(self, advantages: torch.Tensor, num_minibatches: int):
    """
    Multiprocessing compatible mini-batch generator for training the policy of an agent.

    Args:
        advantages (torch.Tensor): Computed advatages.
        num_minibatches (int): Number of minibatches.

    Yields:
        Tuple
    """
    meta_episode_length = self.rewards.shape[0]
    num_meta_episodes = self.rewards.shape[1]

    num_envs_per_batch = num_meta_episodes // num_minibatches
    perm = torch.randperm(num_meta_episodes)

    for start_ind in range(0, num_meta_episodes, num_envs_per_batch):
      obs_batch = []
      recurrent_states_batch = []
      actions_batch = []
      value_preds_batch = []
      return_batch = []
      done_masks_batch = []
      old_action_log_probs_batch = []
      adv_targ = []

      for offset in range(num_envs_per_batch):
        ind = perm[start_ind + offset]
        obs_batch.append(self.obs[:-1, ind])
        actions_batch.append(self.actions[:, ind])
        value_preds_batch.append(self.value_preds[:-1, ind])
        return_batch.append(self.returns[:-1, ind])

        done_masks_batch.append(self.done_masks[:-1, ind])
        old_action_log_probs_batch.append(self.action_log_probs[:, ind])
        adv_targ.append(advantages[:, ind])

        # @todo returns the first hidden state, verify.
        recurrent_states_batch.append(self.recurrent_states[0:1, ind])
        pass

      T, N = meta_episode_length, num_envs_per_batch

      # these are all tensors of size (T, N, -1)
      obs_batch = torch.stack(obs_batch, 1)
      actions_batch = torch.stack(actions_batch, 1)
      value_preds_batch = torch.stack(value_preds_batch, 1)
      return_batch = torch.stack(return_batch, 1)
      done_masks_batch = torch.stack(done_masks_batch, 1)
      old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
      adv_targ = torch.stack(adv_targ, 1)

      # states is just a (N, -1) tensor
      recurrent_states_batch = torch.stack(
        recurrent_states_batch, 1
      ).view(N, -1)

      # flatten the (T, N, ...) tensors to (T * N, ...)
      obs_batch = _flatten(T, N, obs_batch)
      actions_batch = _flatten(T, N, actions_batch)
      value_preds_batch = _flatten(T, N, value_preds_batch)
      return_batch = _flatten(T, N, return_batch)
      done_masks_batch = _flatten(T, N, done_masks_batch)
      old_action_log_probs_batch = _flatten(
        T, N, old_action_log_probs_batch
      )

      adv_targ = _flatten(T, N, adv_targ)

      yield obs_batch, recurrent_states_batch, actions_batch, value_preds_batch, return_batch, done_masks_batch, \
        old_action_log_probs_batch, adv_targ


def _flatten(T: int, N: int, _tensor: torch.Tensor) -> torch.Tensor:
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
