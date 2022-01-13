from ray.rllib.agents.ppo import PPOTrainer


agent = PPOTrainer(config)
agent.restore('/home/radon/ray_results/PPO_HalfCheetah-v2_2022-01-14_00-06-360x72c_5q/checkpoint_000003/checkpoint-3')
agent.evaluate()


