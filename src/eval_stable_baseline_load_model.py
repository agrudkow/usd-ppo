import os

import gym
from stable_baselines3 import PPO

log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Parallel environments
env = gym.make("HalfCheetah-v2")


# Eval model
model = PPO.load('ppo-baseline-half-cheetah-gamma0.99-clip_ration0.1-target_kl0')

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
