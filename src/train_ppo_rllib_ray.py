import argparse
import json

from ray.rllib.agents.ppo import PPOTrainer
from config_rllib_exp import get_rllib_exp_config

def run(save_dir, exp_name):
  """Run grid search for algorithm."""

  for i in range(3, 4, 1):
    for gamma in [0.99, 0.94, 0.89]:
      for clip_ratio in [0.1, 0.2, 0.3]:
        for kl_target in [0.01, 0.05]:
          model_name = f'{exp_name}-gamma{str(gamma)}-\
            clip_ratio{str(clip_ratio)}-target_kl{str(kl_target)}-v{i}'

          model_name = model_name.replace('.', '_')
          print(model_name.center(40, '-'))
          config = get_rllib_exp_config(clip_ratio, gamma, kl_target)
          # Create our RLlib Trainer.
          trainer = PPOTrainer(config=config)

          # Run it for n training iterations. A training iteration includes
          # parallel sample collection by the environment workers as well as
          # loss calculation on the collected batch and a model update.
          for _ in range(1000):
            print(trainer.train())

          trainer.save(f'{save_dir}/{model_name}')
          with open(f'{save_dir}/{model_name}.json', 'w') as f:
            json.dump(config, f, indent=4)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, required=True)
  parser.add_argument('--exp_name', type=str, required=True)
  args = parser.parse_args()
  run(args.save_dir, args.exp_name)
