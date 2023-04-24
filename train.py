import os

from core.training.trainer import ExperimentConfig
from core.training.trainer import Trainer

from core.utils.env_utils import register_custom_envs
register_custom_envs()


if __name__ == '__main__':
  config_path = f'{os.path.dirname(__file__)}/configs/debugging_config.json'
  experiment_config = ExperimentConfig.from_json(config_path)

  # start
  trainer = Trainer(experiment_config)
  trainer.train(False)

  pass
