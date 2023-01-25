
import argparse

import ray
from ray import tune
from ray.rllib.env.policy_server_input import PolicyServerInput

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ip-head',
    type=str,
    default='localhost',
    help='The ip address of the remote server.'
)
parser.add_argument(
    '--base-port',
    type=int,
    default=9900,
    help='The base-port to use. Workers will increment from here.'
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="The number of workers to use. Each worker will create its own listening "
    "socket for incoming SAR data.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default='',
    help="Continue training from a previous run. Checkpoiint should be the full "
    "path to the output directory file according to Abmarl's structure."
)

if __name__ == "__main__":
    args = parser.parse_args()

    # server's address
    server_address = args.ip_head
    server_port = 9900
    print(f'server {args.ip_head}:{server_port}')

    # Policies
    from abmarl.sim.corridor import MultiCorridor
    agents = MultiCorridor().agents
    # NOTE: This ^ is a temporary work around until rllib can get the spaces
    # from the client.
    policies = {
        agent.id: (None, agent.observation_space, agent.action_space, {})
        for agent in agents.values()
    }
    policy_mapping_fn = lambda agent_id: agent_id

    ray.init()

    # configure the trainer
    config = {
        # Use the connector server to generate experiences.
        "input": (
            lambda ioctx: PolicyServerInput(ioctx, server_address, server_port)
        ),
        # Give the observation and action space directly
        "env": None,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
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
