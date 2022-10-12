from abmarl.tools import utils as adu
from abmarl.trainers import DebugTrainer


def run(full_config_path, parameters):
    """Debug the SimulationManagers from the config_file."""

    # Load the experiment as a module
    experiment_mod = adu.custom_import_module(full_config_path)
    title = "DEBUG_" + experiment_mod.params['experiment']['title']

    # Debug the simulation
    sim = experiment_mod.params['experiment']['sim_creator'](
        experiment_mod.params['ray_tune']['config']['env_config']
    )
    trainer = DebugTrainer(
        sim=sim.sim,
        name=title,
        output_dir=experiment_mod.params['ray_tune'].get('local_dir')
    )
    import shutil
    shutil.copy(full_config_path, trainer.output_dir)
    trainer.train(iterations=parameters.episodes, render=parameters.render)
