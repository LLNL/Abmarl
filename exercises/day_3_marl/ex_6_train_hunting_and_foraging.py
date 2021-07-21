# ------------------------------------- #
# --- DSSI 2021 RL Class Exercise 6 --- #
# ------------------------------------- #

# In this exercise, we will setup the configuration file for training the agents,
# paying special attention to the policy setup.

# Import the simulation environment and the agents
from abmarl.sim.examples.forager_hunter import HuntingForagingSim, Forager, Hunter

# Instatiate the foragers and hunters
foragers = {f'forager{i}': Forager(
    id=f'forager{i}',
    agent_view=3,
    attack_range=1,
) for i in range(5)}

hunters = {f'hunter{i}': Hunter(
    id=f'hunter{i}',
    agent_view=2,
    attack_range=1,
) for i in range(2)}

agents = {**hunters, **foragers}

# Instatiate the simulation
sim = HuntingForagingSim(
    12,
    region=20,
    agents=agents
)

# Prepare the environment for use with RLlib
from abmarl.managers import AllStepManager
from abmarl.external.rllib_multiagentenv_wrapper import MultiAgentWrapper
sim = MultiAgentWrapper(AllStepManager(sim))

from ray.tune.registry import register_env
sim_name = "HuntingForaging"
register_env(sim_name, lambda env_config: sim)


# Setup the policies

# Here we have it setup so that every agent on a team trains the same policy.
# Because every agent on the team has the same observation and action space, we can just use
# the specs from one of the agent to define the policies' inputs and outputs.
policies = {
    'foragers': (None, agents['forager0'].observation_space, agents['forager0'].action_space, {}),
    'hunters': (None, agents['hunter0'].observation_space, agents['hunter0'].action_space, {}),
}
def policy_mapping_fn(agent_id):
    if agents[agent_id].team == 2:
        return 'foragers'
    else:
        return 'hunters'

# Parameters
params = {
    'experiment': {
        'title': '{}'.format(sim_name),
        'sim_creator': lambda config=None: sim,
    },
    'ray_tune': {
        'run_or_experiment': 'A2C',
        'checkpoint_freq': 10,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': sim_name,
            'horizon': 200,
            'env_config': {},
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            # --- Parallel ---
            "num_workers": 0,
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
            # "rollout_fragment_length": 200,
            # "batch_mode": "complete_episodes",
            # "train_batch_size": 1000,
        },
    }
}