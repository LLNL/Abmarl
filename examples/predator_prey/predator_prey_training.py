# Setup the simulations
from abmarl.sim.predator_prey import PredatorPreySimulation, Predator, Prey
from abmarl.managers import AllStepManager

region = 6
predators = [Predator(id=f'predator{i}', attack=1) for i in range(2)]
prey = [Prey(id=f'prey{i}') for i in range(7)]
agents = predators + prey

sim_config = {
    'region': region,
    'max_steps': 200,
    'agents': agents,
}
sim_name = 'PredatorPrey'

from abmarl.external.rllib_multiagentenv_wrapper import MultiAgentWrapper
from ray.tune.registry import register_env
sim = MultiAgentWrapper(AllStepManager(PredatorPreySimulation.build(sim_config)))
agents = sim.unwrapped.agents
register_env(sim_name, lambda sim_config: sim)

# Set up heuristic policies
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
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': algo_name,
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 20_000,
        },
        'verbose': 2,
        'config': {
            # --- Simulation ---
            'env': sim_name,
            'env_config': sim_config,
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
            # Number of simulations that each worker starts: int
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
            # 'simple_optimizer': True,
            # "postprocess_inputs": True
        },
    }
}
