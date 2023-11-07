
from abmarl.tools import utils as adu


def _train_rllib(params):
    """
    Train MARL policies with RLlib using parameters dictionary.
    """
    adu.register_env_from_params(params)
    import ray
    from ray import tune
    ray.init()
    tune.run(**params['ray_tune'])
    ray.shutdown()


def train(params):
    """
    Train MARL policies with RLlib using parameters dictionary.

    Args:
        Parameter dictionary that holds all the parameters for training.
    """
    # Copy the configuration module to the output directory
    adu.set_output_directory(params)
    _train_rllib(params)
