
# Setup the environment
from admiral.envs.examples.predator_prey import PredatorPreyEnv, Predator, Prey
from admiral.managers import TurnBasedManager, AllStepManager

region = 6
predators = [Predator(id=f'predator{i}', attack=1) for i in range(2)]
prey = [Prey(id=f'prey{i}') for i in range(7)]
agents = predators + prey

env_config = {
    'region': region,
    'max_steps': 200,
    'agents': agents,
}
env_name = 'PredatorPrey'

from admiral.external.rllib_multiagentenv_wrapper import MultiAgentWrapper
from ray.tune.registry import register_env
env = MultiAgentWrapper(AllStepManager(PredatorPreyEnv.build(env_config)))
agents = env.unwrapped.agents
register_env(env_name, lambda env_config: env)

# Set up heuristic policies
from admiral.pols import RandomAction, HeuristicPolicy

policies = {
    'predator': (None, agents['predator0'].observation_space, agents['predator0'].action_space, {}),
    'prey': (None, agents['prey0'].observation_space, agents['prey0'].action_space, {})
}
def policy_mapping_fn(agent_id):
    if agent_id.startswith('prey'):
        return 'prey'
    else:
        return 'predator'

# Algorithm
algo_name = 'PG'

# Experiment parameters
params = {
    'experiment': {
        'title': '{}'.format('PredatorPrey'),
        'env_creator': lambda config=None: env,
    },
    'ray_tune': {
        'run_or_experiment': algo_name,
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': env_name,
            'env_config': env_config,
            'horizon': 200,
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
            # 'simple_optimizer': True,
            # "postprocess_inputs": True
        },
    }
}

if __name__ == "__main__":
    # Create output directory and save to params
    import os
    import time
    home = os.path.expanduser("~")
    output_dir = os.path.join(home, 'admiral_results/{}_{}'.format(params['experiment']['title'], time.strftime('%Y-%m-%d_%H-%M')))
    params['ray_tune']['local_dir'] = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Copy this configuration file to the output directory
    import shutil
    shutil.copy(os.path.join(os.getcwd(), __file__), output_dir)

    # Initialize and run ray
    import ray
    from ray import tune
    ray.init()
    tune.run(**params['ray_tune'])
    ray.shutdown()
