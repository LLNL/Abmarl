
from abmarl.trainers import DebugTrainer


def debug(params, episodes=1, steps_per_episode=200, render=False, **kwargs):
    """
    Debug the simulation using the parameters.

    Args:
        episodes: The number of episodes to run.
        steps_per_episode: The maximum number of steps to take per episode.
        render: Render the simulation each step.

    Returns:
        The directory where the debug files are saved.
    """
    title = "DEBUG_" + params['experiment']['title']
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

    return trainer.output_dir
