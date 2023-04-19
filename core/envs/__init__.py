from gym.envs.registration import register

register(
    id='Bandit-v0',
    entry_point='core.envs.bandits.bandit_env:BanditEnv',
    max_episode_steps=1,
)

register(
    id='Tabular-v0',
    entry_point='core.envs.mdps.tabular_mdp_env:TabularMDPEnv',
    max_episode_steps=50,
)
