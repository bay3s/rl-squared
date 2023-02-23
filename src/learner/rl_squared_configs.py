from dataclasses import dataclass
from dataclasses_json import dataclass_json
from .constants import *


@dataclass_json
@dataclass
class RLSquaredConfigs:

  trials: int
  epochs_per_trial: int
  episodes_per_epoch: int
  num_value_updares: int
  num_policy_updates: int

  value_lr: float
  policy_lr: float
  discount_rate: float
  epsilon: float

  max_trajectory_length: int
  reset_hidden_state: bool

