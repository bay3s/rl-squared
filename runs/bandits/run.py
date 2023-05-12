import os

import argparse

from rl_squared.training.trainer import ExperimentConfig
from rl_squared.training.trainer import Trainer
from rl_squared.utils.env_utils import register_custom_envs

register_custom_envs()


# @todo uncomment NUM_EPISODES = [10, 100, 500]
NUM_EPISODES = [100]
NUM_ACTIONS = [5, 10, 50]

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
        "--n",
        choices=NUM_EPISODES,
        type=int,
        default=None,
        help=f"Number of episodes, one of [{', '.join([str(n) for n in NUM_EPISODES])}"
        f"].",
    )

    parser.add_argument(
        "--k",
        type=int,
        choices=NUM_ACTIONS,
        default=None,
        help=f"Number of arms, one of [{', '.join([str(n) for n in NUM_ACTIONS])}"
        f"].",
    )

    parser.add_argument(
        "--disable-wandb",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Whether to log the experiment to `wandb`.",
    )

    args = parser.parse_args()

    if args.n is None and not args.run_all:
        raise ValueError(
            f"Unable to infer experiment environment from the inputs, either provide `--env-name` or "
            f"set `--run-all` to `True`"
        )

    if not args.run_all:
        env_names = [f"bandit_n_{args.n}_k_{args.k}"]
    else:
        env_names = []
        for n in NUM_EPISODES:
            for k in NUM_ACTIONS:
                env_names.append(f"bandit_n_{n}_k_{k}")

    for env_name in env_names:
        # config
        config_path = f"{os.path.dirname(__file__)}/configs/{env_name}.json"
        experiment_config = ExperimentConfig.from_json(config_path)

        # train
        trainer = Trainer(experiment_config)
        trainer.train(enable_wandb=not args.disable_wandb)
        pass
