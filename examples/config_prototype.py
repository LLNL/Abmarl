# ---------------------------- #
# --- Setup the simulation --- #
# ---------------------------- #

import sim_class, sim_agents
from abmarl.managers import SimulationManager
from abmarl.external import MultiAgentWrapper
from ray.tune.registry import register_env
sim_config = {
    # Fill in simulation configuration
}

sim_creator = lambda sim_config: MultiAgentWrapper(SimulationManager(sim_class(sim_config)))
sim = sim_creator(sim_config)
sim_name = "SimulationName"
register_env(sim_name, sim_creator)

# -------------------------- #
# --- Setup the policies --- #
# -------------------------- #

from abmarl.pols import HeuristicPolicy


class CustomHeuristicPolicy(HeuristicPolicy):
    """A custom heuristic policy for you the design"""
    def compute_actions(self, obs_batch, *args, **kwargs):
        # return [some_action for _ in obs_batch], [], {}
        return [0 for _ in obs_batch], [], {}


agents = sim.agents
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
        'sim_creator': sim_creator,
    },
    'ray_tune': {
        'run_or_experiment': algo_name,
        'stop': {
            # Stopping criteria
        },
        'config': {
            # --- Simulation ---
            'sim': sim_name,
            'sim_config': sim_config,
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
        },
    }
}
