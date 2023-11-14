
from abmarl.tools import utils as adu


def _train_rllib(params):
    """
    Train MARL policies with RLlib using parameters dictionary.
    """
    from ray import tune
    tune.run(**params['ray_tune'])


def train(params):
    """
    Train MARL policies with RLlib using parameters dictionary.

    Args:
        params: Parameter dictionary that holds all the parameters for training.
    """
    # Copy the configuration module to the output directory
    output_dir = adu.set_output_directory(params)
    _train_rllib(params)
    return output_dir
