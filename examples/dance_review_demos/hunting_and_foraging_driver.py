
# ----------------------------- #
# --- Setup the environment --- #
# ----------------------------- #

# --- Create the agents and the environment --- #

# Import the simulation environment and agents
from admiral.envs.components.examples.hunting_and_foraging import HuntingForagingEnv, ForagingAgent, HuntingAgent

# Instatiate the agents that will operate in this environment. All possible agent
# attributes are listed below.
foragers = {f'forager{i}': ForagingAgent(
    id=f'forager{i}',
    agent_view=5, # Partial Observation Mask: how far away this agent can see other agents.
    team=0, # Which team this agent is on
    move_range=1, # How far the agent can move within a single step.
    min_health=0.0, # If the agent's health falls below this value, it will die.
    max_health=1.0, # Agent's health cannot grow above this value.
    min_harvest=1, # Max amount the agent can harvest in a single step
    max_harvest=1, # Min amount the agent can harvest in a single step
    resource_view=5, # How far away can this agent see resources.
    initial_health=None, # Episode-starting health. If None, then random between min and max health. 
    initial_position=None # Episode-starting position. If None, then random within the region.
) for i in range(7)}

hunters =  {f'hunter{i}': HuntingAgent(
    id=f'hunter{i}', 
    agent_view=2, # Partial Observation Mask: how far away this agent can see other agents.
    team=1, # Which team this agent is on
    move_range=1, # How far the agent can move within a single step.
    min_health=0.0, # If the agent's health falls below this value, it will die.
    max_health=1.0, # Agent's health cannot grow above this value.
    attack_range=1, # How far this agent's attack will reach.
    attack_strength=1.0, # How powerful the agent's attack is.
    attack_accuracy=1.0, # Probability of successful attack
    initial_health=None, # Episode-starting health. If None, then random between min and max health. 
    initial_position=None # Episode-starting position. If None, then random within the region.
) for i in range(2)}

agents = {**foragers, **hunters}

# Instantiate the environment
env = HuntingForagingEnv(
    region=10, # The size of the region, both x and y
    number_of_teams=2, # The number of teams
    min_value=1.0, # The minimum value of the resource before it is marked as depleted, useful if regrow feature is used.
    max_value=1.0, # The max value the resource can attain to, useful if the regrow feature is used
    # attack_norm=np.inf, # The norm to use. Default is np.inf, which means that the attack radius is square box around the agent
    agents=agents # Give the environment the dictionary of agents we created above
)

# --- Prepare the environment for use with RLlib --- #

# Now that you've created the environment, you must wrap it with a simulation manager,
# which controls the timing of the simulation step.
from admiral.managers import AllStepManager # All agents take the step at the same time
env = AllStepManager(env)

# We must wrap the environment with the MultiAgentWrapper so that it
# works with RLlib
from admiral.external.rllib_multiagentenv_wrapper import MultiAgentWrapper
env = MultiAgentWrapper(env)

# Finally we must register the environment with RLlib
from ray.tune.registry import register_env
env_name = "HuntingForaging"
register_env(env_name, lambda env_config: env)


# -------------------------- #
# --- Setup the policies --- #
# -------------------------- #

# Here we have it setup so that every agent on a team trains the same policy.
# Because every agent on the team has the same observation and action space, we can just use
# the specs from one of the agent to define the policies' inputs and outputs.
policies = {
    'foragers': (None, agents['forager0'].observation_space, agents['forager0'].action_space, {}),
    'hunters': (None, agents['hunter0'].observation_space, agents['hunter0'].action_space, {}),
}
def policy_mapping_fn(agent_id):
    if isinstance(agents[agent_id], ForagingAgent):
        return 'foragers'
    else:
        return 'hunters'

# USE FOR DEBUGGING
# print(agents['forager0'].action_space)
# print(agents['forager0'].observation_space)
# print(agents['hunter0'].action_space)
# print(agents['hunter0'].observation_space)
# for agent in agents:
#     print(policy_mapping_fn(agent))
# import sys; sys.exit()


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
        'title': '{}'.format('HuntingForaging'),
    },
    'ray_tune': {
        'run_or_experiment': algo_name,
        'checkpoint_freq': 100,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 20_000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': "HuntingForaging",
            # 'env_config': {},
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            "num_workers": 7,
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
        },
    }
}




if __name__ == "__main__":
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    obs = env.reset()
    env.render(fig=fig)

    for _ in range(100):
        action_dict = {agent.id: agent.action_space.sample() for agent in agents.values() if agent.is_alive}
        _, _, done, _ = env.step(action_dict)
        env.render(fig=fig)
        if done['__all__']:
            break