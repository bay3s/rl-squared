import torch


class FixedCategorical(torch.distributions.Categorical):
    def sample(self) -> torch.Tensor:
        """
        Sample from the categorical distribution.

        Returns:
            torch.Tensor
        """
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Return log probs of actions.

        Args:
            actions (torch.Tensor): Actions to be taken in the environment.

        Returns:
            torch.Tensor
        """
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution.

        Returns:
            torch.Tensor
        """
        return self.probs.argmax(dim=-1, keepdim=True)
