from abmarl.examples import MultiCorridor
from abmarl.managers import TurnBasedManager
from abmarl.external import MultiAgentWrapper
from ray.rllib.examples.policy.random_policy import RandomPolicy

sim = MultiAgentWrapper(TurnBasedManager(MultiCorridor()))

sim_name = "MultiCorridor"
from ray.tune.registry import register_env
register_env(sim_name, lambda sim_config: sim)

ref_agent = sim.sim.agents['agent0']
policies = {
    'corridor_0': (None, ref_agent.observation_space, ref_agent.action_space, {}),
    'corridor_1': (RandomPolicy, ref_agent.observation_space, ref_agent.action_space, {}),
    # 'corridor_1': (None, ref_agent.observation_space, ref_agent.action_space, {}),
}


def policy_mapping_fn(agent_id):
    return f'corridor_{int(agent_id[-1]) % 2}'


# Experiment parameters
params = {
    'experiment': {
        'title': f'{sim_name}',
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': 'PG',
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
