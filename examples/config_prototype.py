
# ----------------------------- #
# --- Setup the environment --- #
# ----------------------------- #

import env_class, env_agents
from admiral.managers import SimulationManager
from admiral.external import MultiAgentWrapper
from ray.tune.registry import register_env
env_config = {
    # Fill in environment_configuration
}

env_creator = lambda env_config: MultiAgentWrapper(SimulationManager(env_class(env_config)))
env = env_creator(env_config)
env_name = "EnvironmentName"
register_env(env_name, env_creator)

# -------------------------- #
# --- Setup the policies --- #
# -------------------------- #

from admiral.pols import HeuristicPolicy


class CustomHeuristicPolicy(HeuristicPolicy):
    """A custom heuristic policy for you the design"""
    def compute_actions(self, obs_batch, *args, **kwargs):
        # return [some_action for _ in obs_batch], [], {}
        return [0 for _ in obs_batch], [], {}


agents = env.agents
policies = {
    'policy_0_name': (None, agents[0].observation_space, agents[0].action_space, {}),
    'policy_1_name': (None, agents[1].observation_space, agents[1].action_space, {}),
    'policy_2_name': (None, agents[2].observation_space, agents[2].action_space, {})
}


def policy_mapping_fn(agent_id):
    pass # Map the agent id to the policy you want that agent to train.


# --------------------------- #
# --- Setup the algorithm --- #
# --------------------------- #

# Full list of supported algorithms here:
# https://docs.ray.io/en/releases-1.2.0/rllib-algorithms.html#rllib-algorithms
algo_name = 'PG'


# ------------------ #
# --- Parameters --- #
# ------------------ #

# List of common ray_tune parameters here:
# https://docs.ray.io/en/releases-1.2.0/rllib-training.html#common-parameters
params = {
    'experiment': {
        'title': '{}'.format('The-title-of-this-experiment'),
        'env_creator': env_creator,
    },
    'ray_tune': {
        'run_or_experiment': algo_name,
        'stop': {
            # Stopping criteria
        },
        'config': {
            # --- Environment ---
            'env': env_name,
            'env_config': env_config,
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
        },
    }
}
