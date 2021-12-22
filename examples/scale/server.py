
import argparse

import ray
from ray import tune
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.examples.custom_metrics_and_callbacks import MyCallbacks
from ray.tune.logger import pretty_print

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    default="DQN",
    choices=["DQN", "PPO"],
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=200,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=500_000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=80.0,
    help="Reward at which we stop training.")
parser.add_argument(
    "--framework",
    choices=["tf", "torch"],
    default="tf",
    help="The DL framework specifier."
)
parser.add_argument(
    '--ip-head',
    type=str,
    default='localhost:9900',
    help='The ip address and port of the remote server.'
)
parser.add_argument(
    "--callbacks-verbose",
    action="store_true",
    help="Activates info-messages for different events on "
    "server/client (episode steps, postprocessing, etc..).")
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. Here,"
    "there is no TensorBoard support.")

# Add this support later
# parser.add_argument(
#     "--no-restore",
#     action="store_true",
#     help="Do not restore from a previously saved checkpoint (location of "
#     "which is saved in `last_checkpoint_[algo-name].out`).")
# parser.add_argument(
#     "--num-workers",
#     type=int,
#     default=2,
#     help="The number of workers to use. Each worker will create "
#     "its own listening socket for incoming experiences.")

if __name__ == "__main__":
    args = parser.parse_args()

    # server's address
    server_address = args.ip_head.split(':')[0]
    server_port = 9900
    print(f'server {server_address}:{server_port}')
    # TODO: Process head node ip without splitting out the port.

    # simulation environment
    from abmarl.sim.corridor import MultiCorridor
    env = MultiCorridor()

    ray.init()

    # configure the trainer
    config = {
        # Use the connector server to generate experiences.
        "input": (
            lambda ioctx: PolicyServerInput(ioctx, server_address, server_port)
        ),
        # Give the observation and action space directly
        "env": None,
        "observation_space": env.observation_space,
        "action_space": env.action_space, # TODO: How to do this for multiagents?
        # Use a single worker process to run the server.
        "num_workers": 0,
        # Disable OPE, since the rollouts are coming from online clients.
        "input_evaluation": [],
        "callbacks": MyCallbacks if args.callbacks_verbose else None,
        "framework": args.framework,
        "log_level": "INFO",
    }

    # Extra config per algorithm
    if args.run == "DQN":
        # Example of using DQN (supports off-policy actions).
        config.update({
            "learning_starts": 100,
            "timesteps_per_iteration": 200,
            "n_step": 3,
        })
        config["model"] = {
            "fcnet_hiddens": [64],
            "fcnet_activation": "linear",
        }
    else:
        # Example of using PPO (does NOT support off-policy actions).
        config.update({
            "rollout_fragment_length": 1000,
            "train_batch_size": 4000,
        })


    # Manual training loop (no Ray tune).
    if args.no_tune:
        if args.run == "DQN":
            trainer = DQNTrainer(config=config)
        else:
            trainer = PPOTrainer(config=config)

        # Serving and training loop.
        ts = 0
        for _ in range(args.stop_iters):
            results = trainer.train()
            print(pretty_print(results))
            if results["episode_reward_mean"] >= args.stop_reward or ts >= args.stop_timesteps:
                break
            ts += results["timesteps_total"]

    # Run with Tune for auto env and trainer creation and TensorBoard.
    else:
        stop = {
            "training_iteration": args.stop_iters,
            "timesteps_total": args.stop_timesteps,
            "episode_reward_mean": args.stop_reward,
        }

        tune.run(
            args.run,
            config=config,
            stop=stop,
            verbose=2
        )