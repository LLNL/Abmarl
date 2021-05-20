
def custom_import_module(full_config_path):
    """
    Import and execute a python file as a module. Useful for import the experiment module and the
    analysis module.

    Parameters:
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
        warnings.warn('Could not find checkpoint_{}. Attempting to load the last checkpoint.'.format(checkpoint_desired))
    
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

def extract_env_and_agents_from_experiment(full_trained_directory, requested_checkpoint):
    # Load the experiment as a module
    # First, we must find the .py file in the directory
    import os
    py_files = [file for file in os.listdir(full_trained_directory) if file.endswith('.py')]
    assert len(py_files) == 1
    full_path_to_config = os.path.join(full_trained_directory, py_files[0])
    experiment_mod = custom_import_module(full_path_to_config)
    # Modify the number of workers in the configuration
    experiment_mod.params['ray_tune']['config']['num_workers'] = 1
    experiment_mod.params['ray_tune']['config']['num_envs_per_worker'] = 1

    checkpoint_dir, checkpoint_value = checkpoint_from_trained_directory(full_trained_directory, requested_checkpoint)
    print(checkpoint_dir)

    # Setup ray
    import ray
    import ray.rllib
    from ray.tune.registry import get_trainable_cls
    ray.init()

    # Get the agent
    alg = get_trainable_cls(experiment_mod.params['ray_tune']['run_or_experiment'])
    agent = alg(
        env=experiment_mod.params['ray_tune']['config']['env'],
        config=experiment_mod.params['ray_tune']['config']    
    )
    agent.restore(os.path.join(checkpoint_dir, 'checkpoint-' + str(checkpoint_value)))

    # Get the environment
    env = experiment_mod.params['experiment']['env_creator'](experiment_mod.params['ray_tune']['config']['env_config'])

    return env, agent

def find_dirs_in_dir(pattern, path):
    """
    Traverse the path looking for directories that match the pattern.

    Return: list of paths that match
    """
    import os, fnmatch
    result = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
