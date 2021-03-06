import argparse
import json

from ray.rllib.agents.ppo import PPOTrainer

if __name__ == '__main__':
  """Evaluate model from RLlib"""

  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_config', type=str, required=True)
  parser.add_argument('--path_to_checkpoint', type=str, required=True)
  args = parser.parse_args()

  with open(args.path_to_config) as data_file:
    config = json.load(data_file)
  config['evaluation_config'] = {"render_env": True}
  agent = PPOTrainer(config)
  agent.restore(args.path_to_checkpoint)
  agent.evaluate()
