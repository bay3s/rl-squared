import os
import sys
import torch

from rl_squared.training.experiment_config import ExperimentConfig
from rl_squared.utils.env_utils import get_render_func, get_vec_normalize, make_vec_envs

sys.path.append("rl_squared")

config_json = f"{os.path.dirname(__file__)}/results/cartpole_v1/config.json"
args = ExperimentConfig.from_json(config_json)

env = make_vec_envs(
    args.env_name,
    args.random_seed + 1000,
    1,
    None,
    None,
    device="cpu",
    allow_early_resets=False,
)

# get a render function
render_func = get_render_func(env)

# we need to use the same statistics for normalization as used in training
actor_critic, obs_rms = torch.load(
    os.path.join(args.checkpoint_dir + "model.pt"), map_location="cpu"
)

vec_norm = get_vec_normalize(env)

if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func("human")

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=True
        )

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if render_func is not None:
        render_func("human")
