
from abmarl.examples import MazeNavigationAgent, MazeNaviationSim
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper

object_registry = {
    'N': lambda n: MazeNavigationAgent(
        id='navigator',
        encoding=1,
        view_range=2,
        render_color='blue',
    ),
    'T': lambda n: GridWorldAgent(
        id='target',
        encoding=3,
        render_color='green'
    ),
    'W': lambda n: GridWorldAgent(
        id=f'wall{n}',
        encoding=2,
        blocking=True,
        render_shape='s'
    )
}

file_name = 'maze.txt'
sim = MultiAgentWrapper(
    AllStepManager(
        MazeNaviationSim.build_sim_from_file(
            file_name,
            object_registry,
            overlapping={1: [3], 3: [1]}
        )
    )
)


sim_name = "MazeNavigation"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)


policies = {
    'navigator': (
        None,
        sim.sim.agents['navigator'].observation_space,
        sim.sim.agents['navigator'].action_space,
        {}
    )
}


def policy_mapping_fn(agent_id):
    return 'navigator'


# Experiment parameters
params = {
    'experiment': {
        'title': f'{sim_name}',
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': 'A2C',
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2000,
        },
        'verbose': 2,
        'config': {
            # --- Simulation ---
            'disable_env_checking': False,
            'env': sim_name,
            'horizon': 200,
            'env_config': {},
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            # "lr": 0.0001,
            # --- Parallelism ---
            # Number of workers per experiment: int
            "num_workers": 7,
            # Number of simulations that each worker starts: int
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
        },
    }
}
