
from admiral.tools import utils as adu

def run(full_trained_directory, full_subscript, parameters):
    """Analyze MARL policies from a saved policy through an analysis script"""
    env, agent = adu.extract_env_and_agents_from_experiment(full_trained_directory, parameters.checkpoint)

    # Load the analysis module and run it
    analysis_mod = adu.custom_import_module(full_subscript)
    analysis_mod.run(env.unwrapped, agent)

    ray.shutdown()