from src.envs.mujoco import (
  AntRandDirecEnv,
  AntRandGoalEnv,
  HalfCheetahRandVelEnv,
  HalfCheetahRandDirecEnv,
  Walker2DRandVelEnv,
  Walker2DRandDirecEnv
)


environments = [
  AntRandDirecEnv,
  AntRandGoalEnv,
  HalfCheetahRandVelEnv,
  HalfCheetahRandDirecEnv,
  Walker2DRandDirecEnv,
  Walker2DRandVelEnv
]

for env_class in environments:
  env = env_class()
  env.render_mode = 'human'
  num_episodes = 10

  for _ in range(num_episodes):
    task = env.sample_tasks(1)[0]
    env.set_task(task)
    env.reset()

    for _ in range(100):
      env.render()
      _, reward, _, _ = env.step(env.action_space.sample())
      pass

  env.close()
  pass

