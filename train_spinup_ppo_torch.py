from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_tf1
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--name', type=str, default='ppo-spinup-pyt-half-cheetah')
    args = parser.parse_args()

    eg = ExperimentGrid(name=args.name)
    eg.add('env_name', 'HalfCheetah-v2', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 1000)
    eg.add('steps_per_epoch', 4000)
    eg.add('gamma', [0.99, 0.94, 0.89])
    eg.add('clip_ratio', [0.1, 0.2, 0.3])
    eg.add('target_kl', [0.01, 0.05])
    eg.run(ppo_tf1, num_cpu=args.cpu)