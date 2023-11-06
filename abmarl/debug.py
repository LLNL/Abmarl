
from abmarl.tools import utils as adu
from abmarl.trainers import DebugTrainer


def debug(params, episodes=1, steps_per_episode=200, render=False):
    """
    Debug the simulation using the parameters.
    
    Args:
        episodes: The number of episodes to run.
        steps_per_episode: The maximum number of steps to take per episode.
        render: Render the simulation each
    """
    title = "DEBUG_" + params['experiment']['title']
    adu.set_output_directory(params)
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
