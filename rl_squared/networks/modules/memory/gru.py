from typing import Tuple

import torch.nn as nn
import torch

from rl_squared.utils.torch_utils import init_gru


class GRU(nn.Module):
    def __init__(self, input_size: int, recurrent_state_size: int):
        """
        Stateful actor for a discrete action space.

        Args:
            input_size (int): State dimensions for the environment.
            recurrent_state_size (int): Size of the recurrent state.
        """
        nn.Module.__init__(self)

        self._gru = init_gru(input_size, recurrent_state_size)
        pass

    def forward(
        self,
        x: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the GRU unit.

        Args:
            x (torch.Tensor): Input to the GRU.
            recurrent_states (torch.Tensor): Recurrent state from the previous forward pass.
            recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.
            device (torch.device): Device on which to transfer tensors.

        Returns:
            Tuple
        """
        if x.size(0) == recurrent_states.size(0):
            if recurrent_state_masks is None:
                recurrent_state_masks = torch.ones(recurrent_states.shape)

            x = x.to(device)
            recurrent_states = recurrent_states.to(device)
            recurrent_state_masks = recurrent_state_masks.to(device)

            x, recurrent_states = self._gru(
                x.unsqueeze(0), (recurrent_states * recurrent_state_masks).unsqueeze(0)
            )

            x = x.squeeze(0)
            recurrent_states = recurrent_states.squeeze(0)

            return x, recurrent_states

        # x is a (T, N, -1) batch from the sampler that has been flattend to (T * N, -1)
        N = recurrent_states.size(0)
        T = int(x.size(0) / N)

        # unflatten
        x = x.view(T, N, x.size(1))

        # masks
        if recurrent_state_masks is None:
            recurrent_state_masks = torch.ones((T, N))
        else:
            recurrent_state_masks = recurrent_state_masks.view(T, N)

        has_zeros = (
            (recurrent_state_masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
        )

        # +1 to correct the recurrent_masks[1:] where zeros are present.
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [T]

        recurrent_state_masks.to(device)
        recurrent_states = recurrent_states.unsqueeze(0)
        outputs = []

        x = x.to(device)
        recurrent_states = recurrent_states.to(device)
        recurrent_state_masks = recurrent_state_masks.to(device)

        for i in range(len(has_zeros) - 1):
            # We can now process steps that don't have any zeros in done_masks together!
            # This is much faster
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, recurrent_states = self._gru(
                x[start_idx:end_idx],
                recurrent_states * recurrent_state_masks[start_idx].view(1, -1, 1),
            )

            outputs.append(rnn_scores)
            pass

        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)

        # flatten
        x = x.view(T * N, -1)
        recurrent_states = recurrent_states.squeeze(0)
        pass

        return x, recurrent_states
