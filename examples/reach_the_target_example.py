
import numpy as np

from abmarl.examples import ReachTheTargetSim, RunningAgent, TargetAgent, BarrierAgent
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper

grid_size = 7
corners = [
    np.array([0, 0], dtype=int),
    np.array([grid_size - 1, 0], dtype=int),
    np.array([0, grid_size - 1], dtype=int),
    np.array([grid_size - 1, grid_size - 1], dtype=int),
]
agents = {
    **{
        f'barrier{i}': BarrierAgent(
            id=f'barrier{i}'
        ) for i in range(10)
    },
    **{
        f'runner{i}': RunningAgent(
            id=f'runner{i}',
            move_range=2,
            view_range=int(grid_size / 2),
            initial_health=1,
            initial_position=corners[i]
        ) for i in range(4)
    },
    'target': TargetAgent(
        view_range=grid_size,
        attack_range=1,
        attack_strength=1,
        attack_accuracy=1,
        initial_position=np.array([int(grid_size / 2), int(grid_size / 2)], dtype=int)
    )
}
overlapping = {
    2: [3],
    3: [1, 2, 3]
}
attack_mapping = {
    2: [3]
}

sim = MultiAgentWrapper(
    AllStepManager(
        ReachTheTargetSim.build_sim(
            7, 7,
            agents=agents,
            overlapping=overlapping,
            attack_mapping=attack_mapping
        )
    )
)


sim_name = "ReachTheTarget"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)


policies = {
    'target': (None, agents['target'].observation_space, agents['target'].action_space, {}),
    'runner': (None, agents['runner0'].observation_space, agents['runner0'].action_space, {}),
}


def policy_mapping_fn(agent_id):
    return 'runner' if agent_id.startswith('runner') else 'target'


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
            'horizon': 20,
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
