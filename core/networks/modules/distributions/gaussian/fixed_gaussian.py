import torch


class FixedGaussian(torch.distributions.Normal):
    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Log probabilities of actions.

        Args:
            actions (torch.Tensor): Actions for which to get the log probabilities.

        Returns:
            torch.Tensor
        """
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> torch.Tensor:
        """
        Get entropy of the distribution.

        Returns:
            torch.Tensor
        """
        return super().entropy().sum(-1)

    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the Normal / Gaussian distribution (which is the same as its mean).

        Returns:
            torch.Tensor
        """
        return self.mean
