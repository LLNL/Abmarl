
# ----------------------------- #
# --- Setup the environment --- #
# ----------------------------- #

import env_class
env_config = {
    # Fill in environment_configuration
}
env, agents= env_class.build(env_config)

from admiral.envs.wrappers import MultiAgentWrapper
from ray.tune.registry import register_env
env_name = "EnvironmentName"
register_env(env_name, lambda env_config: MultiAgentWrapper.wrap(env))


# -------------------------- #
# --- Setup the policies --- #
# -------------------------- #

from admiral.pols import HeuristicPolicy

class CustomHeuristicPolicy(HeuristicPolicy):
    """A custom heuristic policy for you the design"""
    def compute_actions(self, obs_batch, *args, **kwargs):
        # return [some_action for _ in obs_batch], [], {}
        return [0 for _ in obs_batch], [], {}

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

# Full list of supported algorithms here: https://docs.ray.io/en/releases-0.8.5/rllib-algorithms.html
algo_name = 'PG'


# ------------------ #
# --- Parameters --- #
# ------------------ #

# List of common ray_tune parameters here: https://docs.ray.io/en/latest/rllib-training.html#common-parameters
params = {
    'experiment': {
        'title': '{}'.format('The-title-of-this-experiment'),
        'env_creator': lambda config=None: env,
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