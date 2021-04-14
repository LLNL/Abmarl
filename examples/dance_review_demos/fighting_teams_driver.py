
# ----------------------------- #
# --- Setup the environment --- #
# ----------------------------- #

# --- Create the agents and the environment --- #

# Import the simulation environment and agents
from admiral.envs.components.examples.fighting_teams import FightingTeamsEnv, FightingTeamAgent

number_of_teams = 2
region = 15

# Instatiate the agents that will operate in this environment. All possible agent
# attributes are listed below
agents = {f'agent{i}': FightingTeamAgent(
    id=f'agent{i}', # Every agent needs a unique id.
    attack_range=1, # How far this agent's attack will reach.
    attack_strength=1.0, # How powerful the agent's attack is.
    attack_accuracy=1.0, # Probability of successful attack
    team=i%number_of_teams+1, # Every agent is on a team
    move_range=1, # How far the agent can move within a single step.
    min_health=0.0, # If the agent's health falls below this value, it will die.
    max_health=1.0, # Agent's health cannot grow above this value.
    agent_view=region, # Partial Observation Mask: how far away this agent can see other agents.
    initial_health=None, # Episode-starting health. If None, then random between min and max health. 
    initial_position=None # Episode-starting position. If None, then random within the region.
) for i in range(2)}

# Adjust the team to make the learning agents "underdogs"
# import numpy as np
# for agent in agents.values():
#     if np.random.uniform() > 0.6:
#         agent.team = 1
#         agent.initial_position = np.array([np.random.randint(region), 0])
#     else:
#         agent.team = 2
#         agent.initial_position = np.array([np.random.randint(region), region-1])

# Instantiate the environment
env = FightingTeamsEnv(
    number_of_teams=number_of_teams, # Environment must be told the number of teams
    region=region, # Size of the region, in both x and y
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
env_name = "TeamBattle"
register_env(env_name, lambda env_config: env)


# -------------------------- #
# --- Setup the policies --- #
# -------------------------- #

# Here we have it setup so that every agent on a team trains the same policy.
# Because every agent has the same observation and action space, we can just use
# the specs from one of the agent to define the policies' inputs and outputs.
# from admiral.pols import RandomAction
from ray.rllib.examples.policy.random_policy import RandomPolicy
policies = {
    'team1': (None, agents['agent0'].observation_space, agents['agent0'].action_space, {}),
    'team2': (None, agents['agent0'].observation_space, agents['agent0'].action_space, {}),
}
def policy_mapping_fn(agent_id):
    return f'team{agents[agent_id].team}'

# USE FOR DEBUGGING
# print(agents['agent0'].action_space)
# print(agents['agent0'].observation_space)
# # for agent in agents:
# #     print(policy_mapping_fn(agent))
# import sys; sys.exit()


# --------------------------- #
# --- Setup the algorithm --- #
# --------------------------- #

# Full list of supported algorithms here: https://docs.ray.io/en/releases-1.2.0/rllib-algorithms.html
algo_name = 'PG'


# ------------------ #
# --- Parameters --- #
# ------------------ #

# List of common ray_tune parameters here: https://docs.ray.io/en/latest/rllib-training.html#common-parameters
params = {
    'experiment': {
        'title': '{}'.format('TeamBattle'),
    },
    'ray_tune': {
        'run_or_experiment': algo_name,
        'checkpoint_freq': 100_000,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 2_000_000,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': "TeamBattle",
            'horizon': 200,
            # 'env_config': {},
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            "num_workers": 35,
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
            # "rollout_fragment_length": 200,
        },
    }
}


# ---------------------------- #
# --- Random demonstration --- #
# ---------------------------- #

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    obs = env.reset()
    shape_dict={1: 's', 2:'o', 3:'d'}
    env.render(fig=fig, shape_dict=shape_dict)

    import pprint; pprint.pprint(obs['agent0'])

    for _ in range(100):
        action_dict = {agent.id: agent.action_space.sample() for agent in agents.values() if agent.is_alive}
        _, _, done, _ = env.step(action_dict)
        env.render(fig=fig, shape_dict=shape_dict)
        if done['__all__']:
            break