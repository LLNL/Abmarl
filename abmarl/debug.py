
from abmarl.trainers import DebugTrainer

# TODO: make this a common tool among the debug and train
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


def debug(params, episodes=1, steps_per_episode=200, render=False):
    """
    Debug the simulation using the parameters.
    
    Args:
        episodes: The number of episodes to run.
        steps_per_episode: The maximum number of steps to take per episode.
        render: Render the simulation each
    """
    title = "DEBUG_" + params['experiment']['title']
    _set_output_directory(params)
    sim = params['experiment']['sim_creator'](
        params['ray_tune']['config']['env_config']
    )
    trainer = DebugTrainer(
        sim=sim.sim,
        name=title,
        output_dir=params['ray_tune'].get('local_dir')
    )

    trainer.train(
        iterations=episodes,
        render=render,
        horizon=steps_per_episode
    )
