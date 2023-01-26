import os
import shutil

from abmarl.tools import utils as adu

server_top_script = """
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

"""

server_bottom_script="""

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
    tune.run(config)
"""

client_top_script="""
import argparse

from ray.rllib.env.policy_client import PolicyClient

from abmarl.tools import utils as adu

parser = argparse.ArgumentParser()
parser.add_argument(
    '--server_address',
    type=str,
    default='localhost',
    help='The ip address of the remote server.'
)
parser.add_argument(
    '--port',
    type=int,
    default=9900,
    help='The client should connect to this server port.'
)
parser.add_argument(
    "--inference-mode", type=str, default="local", choices=["local", "remote"])


if __name__ == "__main__":
    args = parser.parse_args()

    # Get the simulation from the configuration file
"""

client_bottom_script="""
    sim = experiment_mod.sim

    # policy client
    client = PolicyClient(
        f"http://{args.server_address}:{args.port}",
        inference_mode=args.inference_mode
    )

    # Start data generation
    for i in range(experiment_mod.params['ray_tune']['stop']['episodes_total']):
        eid = client.start_episode(training_enabled=True)
        obs = sim.reset()
        done = {agent: False for agent in obs}
        for j in range(experiment_mod.params['ray_tune']['config']['horizon']):
            action_t = client.get_action(eid, obs)
            action_dict = {
                agent_id: action
                for agent_id, action in action_t.items() if not done[agent_id]
            }
            obs, reward, done, info = sim.step(action_dict)
            client.log_returns(eid, reward, multiagent_done_dict=done, info=info)
            if done['__all__']:
                break
        client.end_episode(eid, obs)
"""

slurm_01 = """#!/bin/bash
#SBATCH --job-name=abmarl-scale-training
"""

slurm_02 = """#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
"""

slurm_03 = """#SBATCH --exclusive
#SBATCH --no-kill
#SBATCH --output="slurm-%j.out"
#SBATCH --ip-isolate yes

# Run command: sbatch client_server.sh

# Source the virtual environment
"""

slurm_04 = """
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

server_node=${nodes_array[0]}
server_node_ip=$(srun --nodes=1 --ntasks=1 -w "$server_node" hostname --ip-address)

# if we detect a space character in the server node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$server_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$server_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  server_node_ip=${ADDR[1]}
else
  server_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $server_node_ip"
fi

echo "Starting server at $server_node"
srun --nodes=1 --ntasks=1 -w "$server_node" --output="slurm-%j-SERVER.out" \
  python3 -u ./server.py --server-address $server_node_ip \
"""

slurm_05 = """
# number of nodes other than the head node
echo "SLURM JOB NUM NODES " $SLURM_JOB_NUM_NODES
clients_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= clients_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting client $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --output="slurm-%j-$node_i.out" \
      python3 -u ./client.py --server-address $server_node_ip \
"""

slurm_06 = """
    sleep 5
done

wait
"""

def run(full_config_path, parameters):
    """
    Generate scripts for running the experiment at scale.
    """
    # Create the directory that will hold the scripts. We name the directory as
    # the full_config_path name appended by "_scale".
    output_dir = os.path.join(
        os.path.dirname(full_config_path),
        f"{os.path.splitext(os.path.basename(full_config_path))[0]}_scale"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the server file
    with open(os.path.join(output_dir, 'server.py'), 'w') as file_writer:
        file_writer.write(server_top_script)
        file_writer.write(f"    experiment_mod = adu.custom_import_module('{full_config_path}')")
        file_writer.write(server_bottom_script)

    # Write the client file
    with open(os.path.join(output_dir, 'client.py'), 'w') as file_writer:
        file_writer.write(client_top_script)
        file_writer.write(f"    experiment_mod = adu.custom_import_module('{full_config_path}')")
        file_writer.write(client_bottom_script)

    # Write the slurm script
    with open(os.path.join(output_dir, 'client_server.sh'), 'w') as file_writer:
        file_writer.write(slurm_01)
        file_writer.write(f"#SBATCH --nodes={parameters.nodes}")
        file_writer.write(slurm_02)
        file_writer.write(f"#SBATCH --time={parameters.time}:00:00")
        file_writer.write(f"#SBATCH --partition={parameters.partition}")
        file_writer.write(slurm_03)
        file_writer.write(f"source {parameters.virtual_env_path}")
        file_writer.write(slurm_04)
        file_writer.write(f"  --base-port {parameters.base_port} &")
        file_writer.write(slurm_05)
        file_writer.write(f"      --port $(({parameters.base_port} + $node_i)) \\")
        file_writer.write(f"      --inference-mode {parameters.inference_mode} &")
        file_writer.write(slurm_06)

# DEBUG:
if __name__ == '__main__':
    run('/Users/rusu1/Abmarl/examples/reach_the_target_example.py')

# def _process_magpie_sbatch(config, full_runnable_config_path):
#     with open(
#         '/usr/tce/packages/magpie/magpie2/submission-scripts/script-sbatch-srun/'
#         'magpie.sbatch-srun-ray', 'r'
#     ) as file_reader:
#         magpie_script = file_reader.readlines()
#     for ndx, line in enumerate(magpie_script):
#         if line.startswith("#SBATCH --nodes"):
#             magpie_script[ndx] = "#SBATCH --nodes={}\n".format(config['nodes'])
#         elif line.startswith("#SBATCH --time"):
#             magpie_script[ndx] = "#SBATCH --time={}:00:00\n".format(config['time_limit'])
#         elif line.startswith("#SBATCH --job-name"):
#             magpie_script[ndx] = "#SBATCH --job-name={}\n".format(config['title'])
#         elif line.startswith("#SBATCH --partition"):
#             magpie_script[ndx] = "#SBATCH --partition=pbatch\n"
#         elif line.startswith('export MAGPIE_PYTHON'):
#             magpie_script[ndx] = 'export MAGPIE_PYTHON="{}"\n'.format(shutil.which('python'))
#         elif line.startswith('export RAY_PATH'):
#             magpie_script[ndx] = 'export RAY_PATH="{}"\n'.format(shutil.which('ray'))
#         elif line.startswith('export RAY_LOCAL_DIR="/tmp/${USER}/ray"'):
#             pass
#             # I cannot specify the output directory becasue it doesn't exist
#             # until the script is actually run!
#         elif line.startswith('export RAY_JOB'):
#             magpie_script[ndx] = 'export RAY_JOB="script"\n'
#         elif line.startswith('# export RAY_SCRIPT_PATH'):
#             magpie_script[ndx] = 'export RAY_SCRIPT_PATH="{}"\n'.format(full_runnable_config_path)
#     with open(
#         os.path.join(
#             os.path.dirname(full_runnable_config_path),
#             f"{config['title']}_magpie.sbatch-srun-ray"), 'w'
#     ) as file_writer:
#         file_writer.writelines(magpie_script)
#     return "    ray.init(address=os.environ['MAGPIE_RAY_ADDRESS'])"


# def run(full_config_path, parameters):
#     """Convert a configuration file to a runnable script, outputting additional
#     scripts as requested."""
#     # Copy the configuration script
#     import os
#     import shutil
#     full_runnable_config_path = os.path.join(
#         os.path.dirname(full_config_path),
#         'runnable_' + os.path.basename(full_config_path)
#     )
#     shutil.copy(full_config_path, full_runnable_config_path)

#     ray_init_line = "    ray.init()"
#     if parameters.magpie:
#         # We need to get two parameters from the experiment configuration. We don't want to load the
#         # entire thing because that is overkill and its costly, so we just read the file and store
#         # a few pieces.
#         with open(full_config_path, 'r') as file_reader:
#             config_items_needed = {
#                 'nodes': parameters.nodes,
#                 'time_limit': parameters.time_limit,
#             }
#             for line in file_reader.readlines():
#                 if line.strip().strip("'").strip('"').startswith('title'):
#                     title = line.split(':')[1].strip().strip(',')
#                     exec("config_items_needed['title'] = {}".format(title))
#                     break
#                     # I'm not worried about executing here becuase the entire module will be
#                     # executed when the script is run.
#         try:
#             ray_init_line = _process_magpie_sbatch(config_items_needed, full_runnable_config_path)
#         except FileNotFoundError:
#             print('Could not find magpie. Is it installed on your HPC system?')

#     # Open the runnable file and write parts to enable runnable
#     with open(full_runnable_config_path, 'a') as file_writer:
#         file_writer.write('\n')
#         file_writer.write(main_if_block_begin)
#         file_writer.write(ray_init_line)
#         file_writer.write(main_if_block_end)
