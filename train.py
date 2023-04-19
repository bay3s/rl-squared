import os

import core.envs
import gym

from core.training.trainer import TrainingArgs
from core.training.trainer import Trainer


if __name__ == '__main__':
  config_path = f'{os.path.dirname(__file__)}/configs/tabular_v0.json'

  args = TrainingArgs.from_json(config_path)
  trainer = Trainer(args)
  trainer.meta_train()
  pass
