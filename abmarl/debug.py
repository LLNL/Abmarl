
from abmarl.trainers import DebugTrainer

def _init_trainer(params):
    """
    Initialize the trainer from the parameters.
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
    return trainer


def _debug_trainer(trainer, parameters):
    trainer.train(
        iterations=parameters.episodes,
        render=parameters.render,
        horizon=parameters.steps_per_episode
    )


def debug(params, episodes=1, steps_per_episode=200, render=False):
    """
    Debug the simulation using the parameters.
    
    Args:
        episodes: The number of episodes to run.
        steps_per_episode: The maximum number of steps to take per episode.
        render: Render the simulation each
    """
    trainer = _init_trainer(params)
    _debug_trainer(
        trainer,
        parameters={
            'episodes': episodes,
            'steps_per_episode': steps_per_episode,
            'render': render
        }
    )
