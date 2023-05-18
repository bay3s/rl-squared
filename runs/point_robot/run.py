import os

import argparse

from rl_squared.training.trainer import ExperimentConfig
from rl_squared.training.trainer import Trainer
from rl_squared.utils.env_utils import register_custom_envs

register_custom_envs()


SUPPORTED_ENVIRONMENTS = ["point_robot_navigation"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="RL-Squared",
        description="Script to run experiments on various meta-learning benchmarks.",
    )

    parser.add_argument(
        "--run-all",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to run all environments, if this is set then the environment parameter is ignored.",
    )

    parser.add_argument(
        "--env-name",
        choices=SUPPORTED_ENVIRONMENTS,
        default=None,
        help=f"Environment to run the experiment on [{', '.join([str(n) for n in SUPPORTED_ENVIRONMENTS])}"
        f"].",
    )

    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint, if any, from which to restart the training run.",
    )

    parser.add_argument(
        "--disable-wandb",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Whether to log the experiment to `wandb`.",
    )

    parser.add_argument(
        "--prod",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Whether this a production run of the experiment.",
    )

    args = parser.parse_args()

    if args.env_name is None and not args.run_all:
        raise ValueError(
            f"Unable to infer experiment environment from the inputs, either provide `--env-name` or "
            f"set `--run-all` to `True`"
        )

    if not args.run_all:
        env_names = [args.env_name]
    else:
        env_names = SUPPORTED_ENVIRONMENTS

    for env_name in env_names:
        # config
        config_path = f"{os.path.dirname(__file__)}/configs/{env_name}.json"
        experiment_config = ExperimentConfig.from_json(config_path)

        # train
        trainer = Trainer(experiment_config, restart_checkpoint=args.from_checkpoint)
        trainer.train(enable_wandb=not args.disable_wandb, is_dev=not args.prod)
        pass
