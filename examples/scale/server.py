
import argparse

import ray
from ray import tune
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.logger import pretty_print

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ip-head',
    type=str,
    default='localhost',
    help='The ip address of the remote server.'
)

if __name__ == "__main__":
    args = parser.parse_args()

    # server's address
    server_address = args.ip_head
    server_port = 9900
    print(f'server {args.ip_head}:{server_port}')

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
    }

    # Run with Tune for auto env and trainer creation and TensorBoard.
    tune.run(
        "A2C",
        config=config,
        stop={
            'episodes_total': 2000,
        },
        verbose=2
    )
