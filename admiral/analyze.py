
from admiral.tools import utils as adu

def _get_checkpoint(full_trained_directory, checkpoint_desired):
    """
    Return the checkpoint directory to load the policy. If checkpoint_desired is specified and
    found, then return that policy. Otherwise, return the last policy.
    """
    checkpoint_dirs = adu.find_dirs_in_dir('checkpoint*', full_trained_directory)

    # Try to load the desired checkpoint
    if checkpoint_desired is not None: # checkpoint specified
        for checkpoint in checkpoint_dirs:
            if checkpoint_desired == int(checkpoint.split('/')[-1].split('_')[-1]):
                return checkpoint, checkpoint_desired
        import warnings
        warnings.warn('Could not find checkpoint_{}. Attempting to load the last checkpoint.'.format(checkpoint_desired))
    
    # Load the last checkpoint
    max_checkpoint_value = 0
    for checkpoint in checkpoint_dirs:
        checkpoint_value = int(checkpoint.split('/')[-1].split('_')[-1])
        if checkpoint_value > max_checkpoint_value:
            max_checkpoint_value = checkpoint_value
            max_checkpoint = checkpoint
    
    return max_checkpoint, max_checkpoint_value

def run(full_trained_directory, full_subscript, parameters):
    """Analyze MARL policies from a saved policy through an analysis script"""

    # Load the experiment as a module
    # First, we must find the .py file in the directory
    import os
    py_files = [file for file in os.listdir(full_trained_directory) if file.endswith('.py')]
    assert len(py_files) == 1
    full_path_to_config = os.path.join(full_trained_directory, py_files[0])
    experiment_mod = adu.custom_import_module(full_path_to_config)
    
    checkpoint_dir, checkpoint_value = _get_checkpoint(full_trained_directory, parameters.checkpoint)
    print(checkpoint_dir)

    # Analyze with ray
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

    # Get the environment. Probably have to do some fancy stuff for the multiagent case
    env = experiment_mod.params['experiment']['env_creator'](experiment_mod.params['ray_tune']['config']['env_config'])

    # Load the analysis module and run it
    analysis_mod = adu.custom_import_module(full_subscript)
    analysis_mod.run(env.unwrapped, agent)

    ray.shutdown()