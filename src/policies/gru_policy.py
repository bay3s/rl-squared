import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUPolicy(nn.Module):

  def __init__(self, num_states: int, num_actions: int, hidden_size: int):
    """
    Initialize a policy to be used with the "River Swim" environment.

    Args:
      num_states (int): The number of states in the environment.
      num_actions (int): Number of actions that can be taken in the environment.
      hidden_size (int): Size of the hidden state.
    """
    super(GRUPolicy, self).__init__()

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.num_states = num_states
    self.num_actions = num_actions
    self.input_dim = self.num_states + self.num_actions + 2
    self.hidden_size = hidden_size

    self.gated_recurrent = nn.GRUCell(self.input_dim, self.hidden_size).to(self.device)
    self.hidden_state = torch.zeros(self.hidden_size).reshape(1, self.hidden_size).to(self.device)
    self.linear_layer = nn.Linear(self.hidden_size, self.num_actions).to(self.device)

    pass

  def reset_hidden_state(self) -> None:
    """
    Resets the hidden state.

    Returns:
      None
    """
    self.hidden_state = torch.zeros(self.hidden_size).reshape(1, self.hidden_size).to(self.device)

  def format_input(self, observation, action, reward, is_done) -> torch.Tensor:
    """
    Formats the input so that we can conduct the forward pass.

    Args:
      observation (torch.Tensor): Observation from the environment.
      action (torch.Tensor): Action taken in the previous step.
      reward (torch.Tensor): Reward from the previous step.
      is_done (torch.Tensor): Whether the episode is complete.

    Returns:
      torch.Tensor
    """
    reward = reward.unsqueeze(1)
    action = F.one_hot(action.long(), num_classes = self.num_actions)
    done = is_done.unsqueeze(1)

    return torch.cat((observation, action, reward, done), dim = 1).to(torch.float).to(self.device)

  def forward(self, observation: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, is_done: torch.Tensor,
              batch: bool = False) -> torch.Tensor:
    """
    Conduct a forward pass through the network.

    Args:
      observation (torch.Tensor): Observation from the environment.
      action (torch.Tensor): Action taken in the previous step.
      reward (torch.Tensor): Reward from the previous step.
      is_done (torch.Tensor): Whether the episode is complete.
      batch (bool): Whether to treat the inputs as a batch.

    Returns:
      torch.Tensor
    """
    x = self.format_input(observation, action, reward, is_done)

    if not batch:
      self.hidden_state = F.relu(self.gru(x, self.hidden_state))
      self.hidden_state = self.hidden_state.to(torch.float).to(self.device)
      self.hidden_state.detach_()

      return self.linear_layer(self.hidden_state)

    # in case the input is in a batched format.
    hidden = torch.cat(x.shape[0] * [self.hidden_state]).detach()
    hidden = self.gated_recurrent(x, hidden)
    hidden = F.relu(hidden)
    hidden = hidden.to(torch.float).to(self.device)
    hidden.detach_()

    return self.linear_layer(hidden)
