import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Parallel environments
env = gym.make("HalfCheetah-v2")
env = Monitor(env, log_dir)
for i in range(5):
    for gamma in [0.99, 0.94, 0.89]:
        for clip_ration in [0.1, 0.2, 0.3]:
            for target_kl in [0.01, 0.05]:
                model_name = f'ppo-baseline-half-cheetah-gamma{str(gamma)}-clip_ration{str(clip_ration)}-target_kl{str(target_kl)}-v{i}'
                model_name = model_name.replace('.', '_')
                print(
                    f'--------------------------------------------------------{model_name}--------------------------------------------'
                )
                model = PPO("MlpPolicy",
                            env,
                            n_epochs=1000,
                            n_steps=4000,
                            gamma=gamma,
                            clip_range=clip_ration,
                            target_kl=target_kl,
                            verbose=1,
                            device='auto',
                            tensorboard_log=f'./logs/{model_name}-tensorboard/')
                model.learn(total_timesteps=4000000)
                model.save(f'./models/{model_name}')
