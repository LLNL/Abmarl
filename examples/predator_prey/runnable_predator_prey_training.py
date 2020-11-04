
# Setup the environment
from admiral.envs.predator_prey import PredatorPrey

region = 6
predators = [{'id': 'predator' + str(i), 'view': region-1} for i in range(1)]
prey = [{'id': 'prey' + str(i), 'view': region-1} for i in range(7)]
agents = predators + prey

env_config = {
    'region': region,
    'max_steps': 200,
    'observation_mode': 'distance',
    'agents': agents,
}
env_name = 'PredatorPrey'

from admiral.envs.wrappers import MultiAgentWrapper
from ray.tune.registry import register_env
env, agents = PredatorPrey.build(env_config)
register_env(env_name, lambda env_config: MultiAgentWrapper.wrap(env))

# Set up heuristic policies
from admiral.pols import RandomAction, HeuristicPolicy
class JustSit(HeuristicPolicy):
    """A policy that has the agents just sit in the same space"""
    def compute_actions(self, obs_batch, *args, **kwargs):
        return [PredatorPrey.Actions.STAY.value for _ in obs_batch], [], {}

import numpy as np
class RunAwayFromSinglePredatorGridObs(HeuristicPolicy):
    """
    A policy that has the agents run away from the predator. This assumes grid
    observations and assumes there is only one predator.
    """
    def compute_actions(self, obs_batch, *args, **kwargs):
        action = []
        for obs in obs_batch:
            my_loc = int(obs.shape[0] / 2)
            predator_x, predator_y = np.where(obs==PredatorPrey.AgentType.PREDATOR.value)[1][0], np.where(obs==PredatorPrey.AgentType.PREDATOR.value)[0][0]
            if abs(predator_x - my_loc) > abs(predator_y - my_loc): # Move in the y direction
                if my_loc > predator_y: # I am below the predator
                    action.append(PredatorPrey.Actions.MOVE_DOWN.value)
                else:
                    action.append(PredatorPrey.Actions.MOVE_UP.value)
            else: # Move in the x direction
                if my_loc > predator_x: # I am to the right of the predator
                    action.append(PredatorPrey.Actions.MOVE_RIGHT.value)
                else:
                    action.append(PredatorPrey.Actions.MOVE_LEFT.value)
        return action, [], {}

class RunAwayFromSinglePredatorDistanceObs(HeuristicPolicy):
    """
    A policy that has the agents run away from the predator. This assumes distance
    observations and assumes there is only one predator.
    """
    def compute_actions(self, obs_batch, *args, **kwargs):
        action = []
        for obs in obs_batch:
            predator_row = np.where(obs[:,2] == PredatorPrey.AgentType.PREDATOR.value)[0][0]
            dist_x, dist_y = obs[predator_row, :2]
            if abs(dist_x) > abs(dist_y): # Move in the y direction
                if dist_y > 0: # I am below the predator
                    action.append(PredatorPrey.Actions.MOVE_DOWN.value)
                else:
                    action.append(PredatorPrey.Actions.MOVE_UP.value)
            else: # Move in the x direction
                if dist_x > 0: # I am to the right of the predator
                    action.append(PredatorPrey.Actions.MOVE_RIGHT.value)
                else:
                    action.append(PredatorPrey.Actions.MOVE_LEFT.value)
        return action, [], {}

policies = {
    'predator': (None, agents['predator0'].observation_space, agents['predator0'].action_space, {}),
    'prey': (RunAwayFromSinglePredatorDistanceObs, agents['prey0'].observation_space, agents['prey0'].action_space, {})
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
            # --- Environment ---
            'env': env_name,
            'env_config': env_config,
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
    ray.init(address=os.environ['MAGPIE_RAY_ADDRESS'])
    tune.run(**params['ray_tune'])
    ray.shutdown()
