import os
import shutil

from abmarl.tools import utils as adu

server_top_level = """
import argparse

from ray import tune
from ray.rllib.env.policy_server_input import PolicyServerInput

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

def run(full_config_path):
    """
    Generate scripts for running the experiment at scale.
    """
    # Load the experiment as a module
    experiment_mod = adu.custom_import_module(full_config_path)
    title = experiment_mod.params['experiment']['title']

    # Copy the configuration module to the output directory
    import os
    import time
    base = experiment_mod.params['ray_tune'].get('local_dir', os.path.expanduser("~"))
    output_dir = os.path.join(
        base, 'abmarl_results/{}_{}'.format(
            title, time.strftime('%Y-%m-%d_%H-%M')
        )
    )
    experiment_mod.params['ray_tune']['local_dir'] = output_dir

    # Train with ray
    import ray
    from ray import tune
    ray.init()
    tune.run(**experiment_mod.params['ray_tune'])
    ray.shutdown()



def _process_magpie_sbatch(config, full_runnable_config_path):
    with open(
        '/usr/tce/packages/magpie/magpie2/submission-scripts/script-sbatch-srun/'
        'magpie.sbatch-srun-ray', 'r'
    ) as file_reader:
        magpie_script = file_reader.readlines()
    for ndx, line in enumerate(magpie_script):
        if line.startswith("#SBATCH --nodes"):
            magpie_script[ndx] = "#SBATCH --nodes={}\n".format(config['nodes'])
        elif line.startswith("#SBATCH --time"):
            magpie_script[ndx] = "#SBATCH --time={}:00:00\n".format(config['time_limit'])
        elif line.startswith("#SBATCH --job-name"):
            magpie_script[ndx] = "#SBATCH --job-name={}\n".format(config['title'])
        elif line.startswith("#SBATCH --partition"):
            magpie_script[ndx] = "#SBATCH --partition=pbatch\n"
        elif line.startswith('export MAGPIE_PYTHON'):
            magpie_script[ndx] = 'export MAGPIE_PYTHON="{}"\n'.format(shutil.which('python'))
        elif line.startswith('export RAY_PATH'):
            magpie_script[ndx] = 'export RAY_PATH="{}"\n'.format(shutil.which('ray'))
        elif line.startswith('export RAY_LOCAL_DIR="/tmp/${USER}/ray"'):
            pass
            # I cannot specify the output directory becasue it doesn't exist
            # until the script is actually run!
        elif line.startswith('export RAY_JOB'):
            magpie_script[ndx] = 'export RAY_JOB="script"\n'
        elif line.startswith('# export RAY_SCRIPT_PATH'):
            magpie_script[ndx] = 'export RAY_SCRIPT_PATH="{}"\n'.format(full_runnable_config_path)
    with open(
        os.path.join(
            os.path.dirname(full_runnable_config_path),
            f"{config['title']}_magpie.sbatch-srun-ray"), 'w'
    ) as file_writer:
        file_writer.writelines(magpie_script)
    return "    ray.init(address=os.environ['MAGPIE_RAY_ADDRESS'])"


def run(full_config_path, parameters):
    """Convert a configuration file to a runnable script, outputting additional
    scripts as requested."""
    # Copy the configuration script
    import os
    import shutil
    full_runnable_config_path = os.path.join(
        os.path.dirname(full_config_path),
        'runnable_' + os.path.basename(full_config_path)
    )
    shutil.copy(full_config_path, full_runnable_config_path)

    ray_init_line = "    ray.init()"
    if parameters.magpie:
        # We need to get two parameters from the experiment configuration. We don't want to load the
        # entire thing because that is overkill and its costly, so we just read the file and store
        # a few pieces.
        with open(full_config_path, 'r') as file_reader:
            config_items_needed = {
                'nodes': parameters.nodes,
                'time_limit': parameters.time_limit,
            }
            for line in file_reader.readlines():
                if line.strip().strip("'").strip('"').startswith('title'):
                    title = line.split(':')[1].strip().strip(',')
                    exec("config_items_needed['title'] = {}".format(title))
                    break
                    # I'm not worried about executing here becuase the entire module will be
                    # executed when the script is run.
        try:
            ray_init_line = _process_magpie_sbatch(config_items_needed, full_runnable_config_path)
        except FileNotFoundError:
            print('Could not find magpie. Is it installed on your HPC system?')

    # Open the runnable file and write parts to enable runnable
    with open(full_runnable_config_path, 'a') as file_writer:
        file_writer.write('\n')
        file_writer.write(main_if_block_begin)
        file_writer.write(ray_init_line)
        file_writer.write(main_if_block_end)
