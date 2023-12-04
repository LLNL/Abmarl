# TODO: Combine this file with the 2-team file

import os

import numpy as np

from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper
from abmarl.examples.sim.traffic_corridor import WallAgent, TargetAgent, TrafficAgent, TrafficCorridorSimulation


grid_1_lane_4_teams = np.array([
    ['g', 'R', 'W', 'W', 'W', 'K', 'b'],
    ['_', '_', '_', '_', '_', '_', '_'],
    ['k', 'B', 'W', 'W', 'W', 'G', 'r'],
])

grid_2_lane_4_teams = np.array([
    ['g', 'R', 'W', 'W', 'W',' W', 'W', 'W', 'W', 'K', 'b'],
    ['_', '_', '_', '_', '_', '_',' _', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_',' _', '_', '_', '_', '_'],
    ['k', 'B', 'W', 'W', 'W',' W', 'W', 'W', 'W', 'G', 'r'],
])


object_registry = {
    'R': lambda n: TrafficAgent(
        id=f'red',
        encoding=1,
        render_color='red',
    ),
    'G': lambda n: TrafficAgent(
        id=f'green',
        encoding=2,
        render_color='green',
    ),
    'B': lambda n: TrafficAgent(
        id=f'blue',
        encoding=3,
        render_color='blue',
    ),
    'K': lambda n: TrafficAgent(
        id=f'black',
        encoding=4,
        render_color='black',
    ),
    'r': lambda n: TargetAgent(
        id=f'red_target',
        encoding=1,
        render_color='red',
        render_shape='s'
    ),
    'g': lambda n: TargetAgent(
        id=f'green_target',
        encoding=2,
        render_color='green',
        render_shape='s'
    ),
    'b': lambda n: TargetAgent(
        id=f'blue_target',
        encoding=3,
        render_color='blue',
        render_shape='s'
    ),
    'k': lambda n: TargetAgent(
        id=f'black_target',
        encoding=4,
        render_color='black',
        render_shape='s'
    ),
    'W': lambda n: WallAgent(
        id=f'wall{n}',
        encoding=5,
        render_shape='s'
    )
}

grid = grid_2_lane_4_teams
sim = MultiAgentWrapper(
    AllStepManager(
        TrafficCorridorSimulation.build_sim_from_array(
            grid,
            object_registry,
            overlapping={1: {1}, 2: {2}, 3: {3}, 4: {4}},
            target_mapping = {
                'red': 'red_target',
                'blue': 'blue_target',
                'green': 'green_target',
                'black': 'black_target',
            }
        )
    )
)

sim_name = "TrafficCoordination"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)


policies = {
    'red': (
        None,
        sim.sim.agents['red'].observation_space,
        sim.sim.agents['red'].action_space,
        {}
    ),
    'blue': (
        None,
        sim.sim.agents['blue'].observation_space,
        sim.sim.agents['blue'].action_space,
        {}
    ),
    'green': (
        None,
        sim.sim.agents['green'].observation_space,
        sim.sim.agents['green'].action_space,
        {}
    ),
    'black': (
        None,
        sim.sim.agents['black'].observation_space,
        sim.sim.agents['black'].action_space,
        {}
    )
}

def policy_mapping_fn(agent_id):
    return agent_id

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
