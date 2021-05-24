
from admiral.envs.examples.corridor import MultiCorridor
from admiral.managers import TurnBasedManager, AllStepManager
from admiral.external import MultiAgentWrapper

env = MultiAgentWrapper(AllStepManager(MultiCorridor()))

env_name = "MultiCorridor"
from ray.tune.registry import register_env
register_env(env_name, lambda env_config: env)


from ray.rllib.examples.policy.random_policy import RandomPolicy
agents = env.unwrapped.agents
policies = {
    'corridor': (None, agents['agent0'].observation_space, agents['agent0'].action_space, {})
}
def policy_mapping_fn(agent_id):
    return 'corridor'

# Experiment parameters
params = {
    'experiment': {
        'title': f'{env_name}',
        'env_creator': lambda config=None: env,
    },
    'ray_tune': {
        'run_or_experiment': 'PG',
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 20_000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': env_name,
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
            # Number of environments that each worker starts: int
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
        },
    }
}
