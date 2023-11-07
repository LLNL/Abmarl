def create_parser(subparsers):
    """Parse the arguments for the train command.

    Returns
    -------
        parser : ArgumentParser
    """
    train_parser = subparsers.add_parser('train', help='Train MARL policies ')
    train_parser.add_argument(
        'configuration', type=str, help='Path to python config file. Include the .py extension.'
    )
    return train_parser


def run(full_config_path):
    from abmarl.tools import utils as adu
    from abmarl.train import _train_rllib
    # Load the experiment as a module
    experiment_mod = adu.custom_import_module(full_config_path)
    params = experiment_mod.params

    # Copy the configuration file
    output_dir = adu.set_output_directory(params)
    import os
    import shutil
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(full_config_path, output_dir)

    # Train the policy
    import ray
    ray.init()
    _train_rllib(params)
    ray.shutdown()
