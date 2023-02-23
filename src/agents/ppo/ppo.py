from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.agents.memory.buffer import Buffer
from src.agents.memory.transition import Transition
from src.agents.ppo.actor_critic import ActorCritic


class PPO:

  def __init__(self, state_dims: int, action_dims: int, actor: nn.Module, critic: nn.Module, actor_lr: float,
               critic_lr: float, epsilon_clipping: float, optimization_steps: int, discount_rate: float):
    """
    Initialize PPO.

    References:
      - Proximal Policy Approximation https://arxiv.org/abs/1707.06347

    Args:
      state_dims (int): Number of dimensions in the state space.
      action_dims (int): Number of dimensions in the action space.
      actor (nn.Module): Actor network for PPO.
      critic (nn.Module): Critic network for PPO.
      actor_lr (float): Learning rate to be used for the actor network.
      critic_lr (float): Learning rate to be used for the critic network.
      epsilon_clipping (float): Epsilon value for the clipped surrogate objective.
      optimization_steps (int): Number of optimization steps for any given policy update.
      discount_rate (float): Rate to be used for discounting the rewards.
    """
    self.state_dims = state_dims
    self.action_dims = action_dims

    self.policy = ActorCritic(state_dims, action_dims, actor, critic)
    self.policy_old = ActorCritic(state_dims, action_dims, actor, critic)
    self.policy_old.load_state_dict(self.policy.state_dict())

    self.optimizer = torch.optim.Adam([
      {'params': self.policy.actor.parameters(), 'lr': actor_lr},
      {'params': self.policy.critic.parameters(), 'lr': critic_lr},
    ])

    self.mse_loss = nn.MSELoss()
    self.epsilon_clipping = epsilon_clipping
    self.optimization_steps = optimization_steps
    self.discount_rate = discount_rate

    self.buffer = Buffer()
    pass

  def to_tensor_buffer(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Re-format the experience and return a tuple of tensors for states, actions, log_probs, and discounted returns.

    Returns:
      Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    states, actions, log_probs, state_values, rewards, is_done = self.buffer.to_tensor()

    states = torch.tensor(np.array(states)).float()
    state_values = torch.tensor(np.array(state_values)).float()
    actions = torch.tensor(np.array(actions)).float()
    log_probs = torch.tensor(np.array(log_probs)).float()
    is_done = torch.tensor(np.array(is_done)).float()

    discounted_returns = []
    ret = 0

    for reward, done in zip(reversed(rewards), reversed(is_done)):
      if done:
        ret = 0

      ret = reward + (self.discount_rate * ret)
      discounted_returns.insert(0, ret)
      pass

    discounted_returns = torch.tensor(np.array(discounted_returns)).float()

    return states, state_values, actions, log_probs, discounted_returns

  def update(self) -> None:
    """
    Make an update to the policy over a specific number of epochs.

    Returns:
      None
    """
    old_states, old_state_values, old_actions, old_log_probs, old_discounted_returns = self.to_tensor_buffer()

    if old_discounted_returns.std() == 0:
      old_discounted_returns = (old_discounted_returns - old_discounted_returns.mean()) / 1e-5
    else:
      old_discounted_returns = (old_discounted_returns - old_discounted_returns.mean()) / old_discounted_returns.std()

    advantages = old_discounted_returns.detach() - old_state_values.detach()
    for epoch in range(self.optimization_steps):
      log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
      ratios = torch.exp(log_probs - old_log_probs)

      min_surrogate = torch.min(
        ratios * advantages,
        torch.clamp(ratios, 1 - self.epsilon_clipping, 1 + self.epsilon_clipping) * advantages
      )

      loss = -1 * min_surrogate + 0.5 * self.mse_loss(state_values, old_discounted_returns) - 0.01 * dist_entropy

      self.optimizer.zero_grad()
      loss.mean().backward()
      self.optimizer.step()
      pass

    # copy new weights into old policy.
    self.policy_old.load_state_dict(deepcopy(self.policy.state_dict()))
    self.buffer.clear()
    pass

  def select_action(self, state: torch.Tensor) -> Tuple:
    """
    Selection action in the discrete action space.

    Args:
      state (torch.Tensor): Current state of the environment.

    Returns:
      Tuple
    """
    with torch.no_grad():
      action, log_prob, state_value = self.policy_old.act(state)

    return action.item(), log_prob.item(), state_value.item()

  def record(self, state, action, log_prob, state_value, reward, is_done: bool) -> None:
    """
    Record a transition in memory.

    Args:
      state (np.ndarray): State in which the action was taken.
      action (int): Action taken by the agent in the current transition.
      log_prob (float): Log probability of the action take.
      state_value (float): State value computed during the transition.
      reward (float): Reward obtained by the agent in the current transition.
      is_done (bool): Whether the episode is done.

    Returns:
      None
    """
    transition = Transition(
      state = state,
      action = action,
      log_prob = log_prob,
      state_value = state_value,
      reward = reward,
      is_done = is_done
    )

    self.buffer.push(transition)
    pass

  def save(self, path: str) -> None:
    """
    Save the old Actor-Critic to the given path.

    Args:
      path (str): Path where to save the network.

    Returns:
      None
    """
    torch.save(self.policy_old.state_dict(), path)
