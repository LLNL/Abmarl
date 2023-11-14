import fnmatch
import os


def custom_import_module(full_config_path):
    """
    Import and execute a python file as a module. Useful for import the experiment module and the
    analysis module.

    Args:
        full_config_path: Full path to the python file.

    Returns: The python file as a module
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("mod", full_config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def checkpoint_from_trained_directory(full_trained_directory, checkpoint_desired):
    """
    Return the checkpoint directory to load the policy. If checkpoint_desired is specified and
    found, then return that policy. Otherwise, return the last policy.
    """
    checkpoint_dirs = find_dirs_in_dir('checkpoint*', full_trained_directory)

    # Try to load the desired checkpoint
    if checkpoint_desired is not None: # checkpoint specified
        for checkpoint in checkpoint_dirs:
            if checkpoint_desired == int(checkpoint.split('/')[-1].split('_')[-1]):
                return checkpoint, checkpoint_desired
        import warnings
        warnings.warn(
            f'Could not find checkpoint_{checkpoint_desired}. Attempting to load the last '
            'checkpoint.'
        )

    # Load the last checkpoint
    max_checkpoint = None
    max_checkpoint_value = 0
    for checkpoint in checkpoint_dirs:
        checkpoint_value = int(checkpoint.split('/')[-1].split('_')[-1])
        if checkpoint_value > max_checkpoint_value:
            max_checkpoint_value = checkpoint_value
            max_checkpoint = checkpoint

    if max_checkpoint is None:
        raise FileNotFoundError("Did not find a checkpoint file in the given directory.")

    return max_checkpoint, max_checkpoint_value


def find_dirs_in_dir(pattern, path):
    """
    Traverse the path looking for directories that match the pattern.

    Return: list of paths that match
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def find_params_from_output_dir(output_dir):
    """
    Find the parameters file from the output directory after a training run.

    Args:
        output_dir: The directory in which to look for the parameters.

    Returns:
        Dictionary of parameters.
    """
    import os
    py_files = [file for file in os.listdir(output_dir) if file.endswith('.py')]
    assert len(py_files) == 1
    full_path_to_config = os.path.join(output_dir, py_files[0])
    experiment_mod = custom_import_module(full_path_to_config)
    return experiment_mod.params


def register_env_from_params(params):
    """
    Register a simulation with RLlib by the simulations title.

    Args:
        params: Dictionary of parameters.
    """
    if type(params['ray_tune']['config']['env']) is str:
        from ray.tune import register_env
        register_env(
            params['experiment']['title'],
            params['experiment']['sim_creator']
        )


def set_output_directory(params):
    """
    Set the output directory in the parameters.

    Args:
        params: Dictionary of parameters

    Returns:
        output_dir: The output directory, also updated in the params.
    """
    import os
    import time
    title = params['experiment']['title']
    base = params['ray_tune'].get('local_dir', os.path.expanduser("~"))
    output_dir = os.path.join(
        base, 'abmarl_results/{}_{}'.format(
            title, time.strftime('%Y-%m-%d_%H-%M')
        )
    )
    params['ray_tune']['local_dir'] = output_dir
    return output_dir
