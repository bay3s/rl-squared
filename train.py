import os

from core.training.trainer import TrainingArgs
from core.training.trainer import Trainer


if __name__ == '__main__':
  config_path = f'{os.path.dirname(__file__)}/configs/cartpole_v1.json'

  args = TrainingArgs.from_json(config_path)
  trainer = Trainer(args)
  trainer.train()
  pass
