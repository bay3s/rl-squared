from gym.envs.registration import register

register(
    id='MultiArmedBandit-v0',
    entry_point='core.envs.bandits.multiarmed_bandit_env:MultiArmedBanditEnv',
    # bandits are a special case where each episode is of length 1
    max_episode_steps=1,
)
