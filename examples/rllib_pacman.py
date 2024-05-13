
from abmarl.examples.sim.pacman import (
    PacmanSimSimple as PacmanSim,
    PacmanAgent, WallAgent, FoodAgent, BaddieAgent
)
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper


object_registry = {
    'P': lambda n: PacmanAgent(
        id='pacman',
        encoding=1,
        view_range=2,
        render_color='yellow',
    ),
    'W': lambda n: WallAgent(
        id=f'wall_{n}',
        encoding=2,
        # blocking=True,
        render_shape='s',
        render_color='b'
    ),
    'F': lambda n: FoodAgent(
        id=f'food_{n}',
        encoding=3,
        render_color='white'
    ),
    'B': lambda n: BaddieAgent(
        id=f'baddie_{n}',
        encoding=4,
        render_color='r'
    ),
}
extra_agents = {
    'pacman': PacmanAgent(
        id='pacman',
        encoding=1,
        view_range=2,
        render_color='yellow',
    )
}
sim = MultiAgentWrapper(
    AllStepManager(
        PacmanSim.build_sim_from_array(
            PacmanSim.example_grid,
            object_registry,
            # extra_agents=extra_agents,
            states={'PositionState', 'OrientationState', 'HealthState'},
            observers={'AbsoluteEncodingObserver'},
            overlapping={1: {3, 4}, 4: {3, 4}},
            reward_scheme={
                'bad_move': 0,
                'entropy': -0.01,
                'eat_food': 0.2,
                'die': -1
            }
        )
    )
)

sim_name = "Pacman"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)

policies = {
    'pacman': (
        None, sim.sim.agents['pacman'].observation_space, sim.sim.agents['pacman'].action_space, {}
    ),
    'baddie': (
        None,
        sim.sim.agents['baddie_0'].observation_space,
        sim.sim.agents['baddie_0'].action_space, {}
    ),
}


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id.startswith('pacman'):
        return 'pacman'
    else:
        return 'baddie'


params = {
    'experiment': {
        'title': f'{sim_name}',
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': 'PPO',
        'checkpoint_freq': 1,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 500,
        },
        'verbose': 2,
        'storage_path': 'output_dir',
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
                'policies_to_train': ['pacman'],
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
