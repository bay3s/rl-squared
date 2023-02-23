from abc import ABC


class BaseRecurrentPolicy(ABC):

  def reset_hidden(self) -> None:
    """
    Reset the hiddent state of the recurrent policy.

    Returns:
      None
    """
    raise NotImplementedError
