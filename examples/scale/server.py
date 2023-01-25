
import argparse

from ray import tune
from ray.rllib.env.policy_server_input import PolicyServerInput

from abmarl.tools import utils as adu

parser = argparse.ArgumentParser()
parser.add_argument(
    '--server_address',
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
# TODO: Must do abmarl-360 first
# parser.add_argument(
#     "--restore",
#     type=str,
#     default='',
#     help="Continue training from a previous run. Restore should be the full "
#     "path to the output directory file according to Abmarl's structure."
# )

if __name__ == "__main__":
    args = parser.parse_args()

    def _input(ioctx):
        # We are remote worker or we are local worker with num_workers=0:
        # Create a PolicyServerInput.
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                args.server_address,
                args.base_port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        # No InputReader (PolicyServerInput) needed.
        else:
            return None

    # TODO: Scale should write out the full config path
    experiment_mod = adu.custom_import_module(full_config_path)

    # Policies
    # TODO: Get the policies from the configuration file instead
    # NOTE: This is a temporary work around until rllib can get the spaces
    # from the client.
    from abmarl.examples.sim.multi_corridor import MultiCorridor
    agents = MultiCorridor().agents
    policies = {
        'target': (None, agents['target'].observation_space, agents['target'].action_space, {}),
        'runner': (None, agents['runner0'].observation_space, agents['runner0'].action_space, {}),
    }

    def policy_mapping_fn(agent_id):
        return 'runner' if agent_id.startswith('runner') else 'target'


    # Trainer config.
    # Get the config from the configuration file
    config = {
        # Use the `PolicyServerInput` to generate experiences.
        "input": _input,
        # Disable OPE, since the rollouts are coming from online clients.
        "off_policy_estimation_methods": {},
        **experiment_mod.params['ray_tune']
    }

    # TODO: Must do abmarl-360 first
    # # Attempt to restore from checkpoint, if possible.
    # if args.restore and os.path.exists(args.restore):
    #     restore_from_path = open(args.restore).read()
    #     # TODO: We need to dig into the directory to get the checkpoint from which to restore
    # else:
    #     restore_from_path = None

    # Run with Tune for auto env and trainer creation and TensorBoard.
    tune.run(**experiment_mod.params['ray_tune'])
