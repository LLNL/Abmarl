
import numpy as np

from abmarl.examples import ResourceAgent, PreyAgent, PredatorAgent, PredatorPreyResourcesSim
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper

# 2 predators
# 5 prey
# 11 resources

resources = {
    f'resource_{i}': ResourceAgent(id=f'resource_{i}') for i in range(11)
}
prey = {
    f'prey_{i}': PreyAgent(id=f'prey_{i}') for i in range(5)
}
predators = {
    f'predator_{i}': PredatorAgent(id=f'predator_{i}') for i in range(2)
}
agents = {**resources, **prey, **predators}

overlap_map = {
    1: {1},
    2: {2},
    3: {3},
}
attack_map = {
    2: {1},
    3: {2}
}
sim = MultiAgentWrapper(
    AllStepManager(
        PredatorPreyResourcesSim.build_sim(
            8, 8,
            agents=agents,
            overlapping=overlap_map,
            attack_mapping=attack_map,
            states={'PositionState', 'HealthState'},
            observers={'PositionCenteredEncodingObserver'},
            dones={'ActiveDone, TargetEncodingInactiveDone'}
        )
    )
)


sim_name = "PredatorPreyResources"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)


policies = {
    'prey': (None, prey['prey_0'].observation_space, prey['prey_0'].action_space, {}),
    'predator': (
        None, predators['predator_0'].observation_space, predators['predator_0'].action_space, {}
    ),
}


def policy_mapping_fn(agent_id):
    if agents[agent_id].encoding == 1:
        return 'prey'
    if agents[agent_id].encoding == 2:
        return 'predator'


# Experiment parameters
params = {
    'experiment': {
        'title': f'{sim_name}',
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': 'PPO',
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 20_000,
        },
        'verbose': 2,
        'local_dir': 'output_dir',
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
            # --- Parallelism ---
            # Number of workers per experiment: int
            "num_workers": 7,
            # Number of simulations that each worker starts: int
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
        },
    }
}
