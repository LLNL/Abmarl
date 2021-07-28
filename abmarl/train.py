from abmarl.tools import utils as adu


def run(full_config_path):
    """Train MARL policies using the config_file."""

    # Load the experiment as a module
    experiment_mod = adu.custom_import_module(full_config_path)
    title = experiment_mod.params['experiment']['title']

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

    # Train with ray
    import ray
    from ray import tune
    ray.init()
    tune.run(**experiment_mod.params['ray_tune'])
    ray.shutdown()
