import torch


class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions: torch.Tensor):
        """
        Returns the log probabilities for the Bernoulli.

        Args:
            actions (torch.Tensor): Actions for which to get the log probabilities.

        Returns:
            torch.Tensor
        """
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of the distribution.

        Returns:
            torch.Tensor
        """
        return super().entropy().sum(-1)

    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution.

        Returns:
            torch.Tensor
        """
        return torch.gt(self.probs, 0.5).float()
