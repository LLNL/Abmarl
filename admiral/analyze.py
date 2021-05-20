
from admiral.tools import utils as adu

def run(full_trained_directory, full_subscript, parameters):
    """Analyze MARL policies from a saved policy through an analysis script"""

    # Load the experiment as a module
    # First, we must find the .py file in the directory
    import os
    py_files = [file for file in os.listdir(full_trained_directory) if file.endswith('.py')]
    assert len(py_files) == 1
    full_path_to_config = os.path.join(full_trained_directory, py_files[0])
    experiment_mod = adu.custom_import_module(full_path_to_config)
    
    checkpoint_dir, checkpoint_value = adu.checkpoint_from_trained_directory(full_trained_directory, parameters.checkpoint)
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