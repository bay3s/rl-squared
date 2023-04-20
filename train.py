import os

import wandb

from core.training.trainer import ExperimentConfig
from core.training.trainer import Trainer

from core.utils.env_utils import register_custom_envs
register_custom_envs()


if __name__ == '__main__':
  wandb.login()

  config_path = f'{os.path.dirname(__file__)}/configs/tabular_v0.json'
  args = ExperimentConfig.from_json(config_path)

  # start
  trainer = Trainer(args)
  trainer.train()

  pass
