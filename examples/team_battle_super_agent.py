
import numpy as np

from abmarl.examples import BattleAgent, TeamBattleSim
from abmarl.sim.wrappers import SuperAgentWrapper
from abmarl.managers import AllStepManager
from abmarl.external import MultiAgentWrapper


colors = ['red', 'blue', 'green', 'gray']
positions = [np.array([1, 1]), np.array([1, 6]), np.array([6, 1]), np.array([6, 6])]
agents = {
    f'agent{i}': BattleAgent(
        id=f'agent{i}',
        encoding=i % 4 + 1,
        render_color=colors[i % 4],
        initial_position=positions[i % 4],
    ) for i in range(24)
}
overlap_map = {
    1: [1],
    2: [2],
    3: [3],
    4: [4]
}
attack_map = {
    1: [2, 3, 4],
    2: [1, 3, 4],
    3: [1, 2, 4],
    4: [1, 2, 3]
}
super_agent_mapping = {
    'red': [agent.id for agent in agents.values() if agent.encoding == 1],
    'blue': [agent.id for agent in agents.values() if agent.encoding == 2],
    'green': [agent.id for agent in agents.values() if agent.encoding == 3],
    'gray': [agent.id for agent in agents.values() if agent.encoding == 4],
}


sim_ = AllStepManager(
    SuperAgentWrapper(
        TeamBattleSim.build_sim(
            8, 8,
            agents=agents,
            overlapping=overlap_map,
            attack_mapping=attack_map
        ),
        super_agent_mapping=super_agent_mapping,
    )
)
sim = MultiAgentWrapper(sim_)


sim_name = "TeamBattle"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)


policies = {
    'red': (None, sim_.agents['red'].observation_space, sim_.agents['red'].action_space, {}),
    'blue': (None, sim_.agents['blue'].observation_space, sim_.agents['blue'].action_space, {}),
    'green': (None, sim_.agents['green'].observation_space, sim_.agents['green'].action_space, {}),
    'gray': (None, sim_.agents['gray'].observation_space, sim_.agents['gray'].action_space, {}),
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
