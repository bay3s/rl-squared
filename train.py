import os

from core.training.trainer import ExperimentConfig
from core.training.trainer import Trainer

from core.utils.env_utils import register_custom_envs

register_custom_envs()


if __name__ == "__main__":
    config_path = f"{os.path.dirname(__file__)}/configs/bandit_v0.json"
    experiment_config = ExperimentConfig.from_json(config_path)

    # start
    trainer = Trainer(experiment_config)
    trainer.train(enable_wandb=False)
    pass
