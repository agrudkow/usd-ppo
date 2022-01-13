from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import json

for i in range(5):
    for gamma in [0.99, 0.94, 0.89]:
        for clip_ratio in [0.1, 0.2, 0.3]:
            for kl_target in [0.01, 0.05]:
                model_name = f'ppo-rllib-half-cheetah-gamma{str(gamma)}-clip_ration{str(clip_ratio)}-target_kl{str(kl_target)}-v{i}'
                model_name = model_name.replace('.', '_')
                print(
                    f'--------------------------------------------------------{model_name}--------------------------------------------'
                )
                # Configure the algorithm.
                config = {
                    # Environment (RLlib understands openAI gym registered strings).
                    "env": "HalfCheetah-v2",
                    # Use 2 environment workers (aka "rollout workers") that parallelly
                    # collect samples from their own environment clone(s).
                    "num_workers": 10,
                    # Change this to "framework: torch", if you are using PyTorch.
                    # Also, use "framework: tf2" for tf2.x eager execution.
                    "framework": "tf",
                    # Tweak the default model provided automatically by RLlib,
                    # given the environment's observation- and action spaces.
                    "model": {
                        "fcnet_hiddens": [64, 64],
                        "fcnet_activation": "relu",
                    },
                    "train_batch_size": 4000,
                    "sgd_minibatch_size": 64,
                    "clip_param": clip_ratio,
                    "gamma": gamma,
                    "kl_target": kl_target,
                    # Set up a separate evaluation worker set for the
                    # `trainer.evaluate()` call after training (see below).
                    "evaluation_num_workers": 1,
                    # Only for evaluation runs, render the env.
                    "evaluation_config": {
                        "render_env": False,
                    }
                }

                # Create our RLlib Trainer.
                trainer = PPOTrainer(config=config)

                # Run it for n training iterations. A training iteration includes
                # parallel sample collection by the environment workers as well as
                # loss calculation on the collected batch and a model update.
                for _ in range(3):
                    print(trainer.train())

                checkpoint_path = trainer.save(f'./models/rllib/{model_name}')
                with open(f'./models/rllib/{model_name}.json', 'w') as f:
                    json.dump(config, f,  indent=4)
                # Evaluate the trained Trainer (and render each timestep to the shell's
                # output).
                # trainer.evaluate()


