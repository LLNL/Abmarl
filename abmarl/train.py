
def _set_output_directory(params):
    """
    Set the output directory in the parameters.
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


def _train_rllib(params):
    """
    Train MARL policies with RLlib using parameters dictionary.
    """
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
    output_dir = _set_output_directory(params)
    _train_rllib(params)
