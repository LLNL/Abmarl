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


def run(full_config_path):
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
