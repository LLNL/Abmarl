
import numpy as np

from abmarl.sim.gridworld.examples.team_battle_example import BattleAgent, TeamBattleSim
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
        initial_position=positions[i % 4]
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
super_agent_mapping={
    'red': [agent.id for agent in agents.values() if agent.encoding == 1],
    'blue': [agent.id for agent in agents.values() if agent.encoding == 2],
    'green': [agent.id for agent in agents.values() if agent.encoding == 3],
    'gray': [agent.id for agent in agents.values() if agent.encoding == 4],
}
null_obs = {'grid': -2 * np.ones((7,7), dtype=int)}

sim = MultiAgentWrapper(
    AllStepManager(
        SuperAgentWrapper(
            TeamBattleSim.build_sim(
                8, 8,
                agents=agents,
                overlapping=overlap_map,
                attack_mapping=attack_map
            ), 
            super_agent_mapping=super_agent_mapping,
            null_obs={
                agent_id: null_obs for agent_id in agents
            }
        )
    )
)

sim_name = "TeamBattle"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)


policies = {
    'red': (None, agents['agent0'].observation_space, agents['agent0'].action_space, {}),
    'blue': (None, agents['agent1'].observation_space, agents['agent1'].action_space, {}),
    'green': (None, agents['agent2'].observation_space, agents['agent2'].action_space, {}),
    'gray': (None, agents['agent3'].observation_space, agents['agent3'].action_space, {}),
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
            'disable_env_checking': True,
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
            "num_workers": 0,
            # Number of simulations that each worker starts: int
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
        },
    }
}
