config_rllib_exp = {
              # Environment (RLlib understands openAI gym registered strings).
              "env": "HalfCheetah-v2",
              # Use 2 environment workers (aka "rollout workers") that parallelly
              # collect samples from their own environment clone(s).
              "num_workers": 2,
              # Change this to "framework: torch", if you are using PyTorch.
              # Also, use "framework: tf2" for tf2.x eager execution.
              "framework": "tf",
              # Tweak the default model provided automatically by RLlib,
              # given the environment's observation- and action spaces.
              "model": {
                  "fcnet_hiddens": [64, 64],
                  "fcnet_activation": "tanh",
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