from abmarl.tools import utils as adu
from abmarl.trainers import DebugTrainer


def run(full_config_path, parameters):
    """Debug the SimulationManagers from the config_file."""

    # Load the experiment as a module
    experiment_mod = adu.custom_import_module(full_config_path)
    title = "DEBUG_" + experiment_mod.params['experiment']['title']

    # Copy the configuration module to the output directory
    import os
    import shutil
    import time
    base = experiment_mod.params['ray_tune'].get('local_dir', os.path.expanduser("~"))
    output_dir = os.path.join(
        base, 'abmarl_results/{}_{}'.format(
            title, time.strftime('%Y-%m-%d_%H-%M')
        )
    )
    experiment_mod.params['ray_tune']['local_dir'] = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(full_config_path, output_dir)

    # Debug the simulation
    sim = experiment_mod.params['experiment']['sim_creator'](
        experiment_mod.params['ray_tune']['config']['env_config']
    )
    trainer = DebugTrainer(sim=sim.sim, output_dir=output_dir)
    trainer.train(iterations=parameters.episodes, render=parameters.render)
