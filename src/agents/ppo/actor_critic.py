from typing import Tuple

import torch.nn as nn
import torch
from torch.distributions import Categorical


class ActorCritic(nn.Module):

  def __init__(self, state_dims: int, action_dims: int, actor: nn.Module, critic: nn.Module):
    """
    Actor-Critic for a discrete action space.

    Args:
      state_dims (int): State dimensions for the environment.
      action_dims (int): Action dimensions for the environment.
      actor (nn.Module): Neural net to be used as the actor.
      critic (nn.Module): Neural net to be used as the critic.
    """
    super(ActorCritic, self).__init__()

    self.state_dims = state_dims
    self.action_dims = action_dims

    self.actor = actor
    self.critic = critic
    pass

  def act(self, state: torch.Tensor) -> Tuple:
    """
    Given a state return the action to take, log probability of said action, and the current state value
    computed by the critic.

    Args:
      state (torch.Tensor): The state in which to take an action.

    Returns:
      Tuple
    """
    probs = self.actor(state)
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    state_value = self.critic(state)

    return action, log_prob, state_value

  def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple:
    """
    Given a state and an action - evaluate the value of the state, log probabilities of the action, and distribution
    entropy.

    Args:
      state (torch.Tensor): The state in which to take an action.
      action (torch.Tensor): The action to take given in the current state.

    Returns:
      Tuple
    """
    probs = self.actor(state)
    dist = Categorical(probs)
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    state_values = self.critic(state)

    return log_prob, torch.squeeze(state_values), entropy

  def forward(self) -> None:
    """
    Forward pass for the network, in this case not implemented.

    Returns:
      None
    """
    raise NotImplementedError
