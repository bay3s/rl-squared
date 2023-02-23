from typing import List, Tuple

from src.memory.transition import Transition


class Buffer:

  def __init__(self) -> None:
    """
    Initialize a rollout.
    """
    self._transitions = list()
    pass

  def push(self, transition: Transition) -> Transition:
    """
    Add the results of an episode step to the memory.

    :param transition: the episode step to add to the rollout.

    :return: None | Transition
    """
    self._transitions.append(transition)

    return transition

  @property
  def transitions(self) -> List[Transition]:
    """
    Return the list of episode steps in the current memory.
    :return: List[Transition]
    """
    return self._transitions

  def __len__(self) -> int:
    """
    Returns the number of transitions in the replay memory at the moment.
    :return: int
    """
    return len(self._transitions)

  def clear(self) -> None:
    """
    Reset the rollout buffer to its original state.

    :return: None
    """
    self._transitions = list()

  def to_tensor(self) -> Tuple:
    """
    Returns tensors containing data from the transitions

    Returns:
      Tuple
    """
    transitions = list(zip(*self.transitions))

    states, actions, log_probs, state_values, rewards, is_done = transitions[0], transitions[1], transitions[2], \
        transitions[3], transitions[4], transitions[5]

    return states, actions, log_probs, state_values, rewards, is_done
