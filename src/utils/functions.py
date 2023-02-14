"""
The file defines utility functions that can be brooadly used for experiments and algorithm implementations.
"""


def ema(data: list, alpha: float = 0.9) -> list:
  """
  Computes the exponential moving averages (EMA) for a given list and returns the result.

  Args:
    data (list): Input data for which to compute EMAs.
    alpha (float): The EMA factor.

  Returns:
    list
  """
  averages = list()

  for i, value in enumerate(data):
    if i == 0:
      averages.append(value)
      continue

    previous = averages[i - 1]
    averages.append((1 - alpha) * value + alpha * previous)

  return averages
