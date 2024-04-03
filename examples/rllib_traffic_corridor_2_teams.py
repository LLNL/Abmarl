
import numpy as np

from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper
from abmarl.examples.sim.traffic_corridor import WallAgent, TargetAgent, TrafficAgent, \
    TrafficCorridorSimulation

grid = np.array([
    ['G', 'W', 'W', 'W', 'R'],
    ['r', '_', '_', '_', 'g'],
    ['G', 'W', 'W', 'W', 'R'],
])


object_registry = {
    'R': lambda n: TrafficAgent(
        id=f'red{n}',
        encoding=1,
        render_color='red',
    ),
    'G': lambda n: TrafficAgent(
        id=f'green{n}',
        encoding=2,
        render_color='green',
    ),
    'r': lambda n: TargetAgent(
        id='red_target',
        encoding=1,
        render_color='red',
        render_shape='s'
    ),
    'g': lambda n: TargetAgent(
        id='green_target',
        encoding=2,
        render_color='green',
        render_shape='s'
    ),
    'W': lambda n: WallAgent(
        id=f'wall{n}',
        encoding=3,
        render_shape='s'
    )
}


sim = MultiAgentWrapper(
    AllStepManager(
        TrafficCorridorSimulation.build_sim_from_array(
            grid,
            object_registry,
            overlapping={1: {1}, 2: {2}},
            states={"PositionState"},
            dones={"TargetAgentOverlapDone"},
            observers={'PositionCenteredEncodingObserver'},
            target_mapping={
                'red0': 'red_target',
                'red1': 'red_target',
                'green0': 'green_target',
                'green1': 'green_target',
            }
        ),
        randomize_action_input=True,
    )
)

sim_name = "TrafficCoordination"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)

policies = {
    'red': (
        None,
        sim.sim.agents['red0'].observation_space,
        sim.sim.agents['red0'].action_space,
        {}
    ),
    'green': (
        None,
        sim.sim.agents['green0'].observation_space,
        sim.sim.agents['green0'].action_space,
        {}
    ),
}


def policy_mapping_fn(agent_id):
    if agent_id.startswith('red'):
        return "red"
    elif agent_id.startswith('green'):
        return "green"


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
            'episodes_total': 20_000,
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
