
import os

import numpy as np

from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper
from abmarl.examples.sim.traffic_corridor import WallAgent, TargetAgent, TrafficAgent, TrafficCorridorSimulation

grid_1_lane_2_teams = np.array([
    ['G', 'W', 'W', 'W', 'R'],
    ['r', '_', '_', '_', 'g'],
    ['G', 'W', 'W', 'W', 'R'],
])

grid_2_lane_2_teams = np.array([
    ['_', 'W', 'W', 'W',' W', 'W', 'W', 'W', 'g'],
    ['G', '_', '_', '_',' _', '_', '_', '_', 'R'],
    ['G', '_', '_', '_',' _', '_', '_', '_', 'R'],
    ['r', 'W', 'W', 'W',' W', 'W', 'W', 'W', '_'],
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
    'W': lambda n: WallAgent(
        id=f'wall{n}',
        encoding=3,
        render_shape='s'
    )
}


grid = grid_1_lane_2_teams
sim = MultiAgentWrapper(
    AllStepManager(
        TrafficCorridorSimulation.build_sim_from_array(
            grid,
            object_registry,
            overlapping={1: {1}, 2: {2}},
            # TODO: Don't know the agent's id's beforehand, how to create better mapping?
            # See if I can use mapping by encoding....
            state={"PositionState"},
            dones={"TargetAgentDone"},
            observers={'PositionCenteredEncodingObserver'},
            target_mapping = {
                'red9': 'red_target',
                'red11': 'red_target',
                'green8': 'green_target',
                'green10': 'green_target',
            }
        ),
        randomize_action_input=True,
    )
)

sim_name = "TrafficCoordination"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)

red_agent_id = next(iter([agent_id for agent_id in sim.sim.agents if agent_id.startswith('red')]))
green_agent_id = next(iter([agent_id for agent_id in sim.sim.agents if agent_id.startswith('green')]))
policies = {
    'red': (
        None,
        sim.sim.agents[red_agent_id].observation_space,
        sim.sim.agents[red_agent_id].action_space,
        {}
    ),
    'green': (
        None,
        sim.sim.agents[green_agent_id].observation_space,
        sim.sim.agents[green_agent_id].action_space,
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
