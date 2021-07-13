
# ----------------------------- #
# --- Setup the environment --- #
# ----------------------------- #

# --- Create the agents and the environment --- #

# Import the simulation environment and agents
from abmarl.sim.components.examples.hunting_and_foraging import HuntingForagingEnv, \
    HuntingForagingAgent, FoodAgent

# Instatiate the agents that will operate in this environment. All possible agent
# attributes are listed below.

# Food agents are not really agents in the RL sense. They're just entites for the
# foragers to eat.
food = {f'food{i}': FoodAgent(id=f'food{i}', team=1) for i in range(12)}

# Foragers try to eat all the food agents before they die
foragers = {f'forager{i}': HuntingForagingAgent(
    id=f'forager{i}',
    agent_view=3, # Partial Observation Mask: how far away this agent can see other agents.
    team=2, # Which team this agent is on
    move_range=1, # How far the agent can move within a single step.
    min_health=0.0, # If the agent's health falls below this value, it will die.
    max_health=1.0, # Agent's health cannot grow above this value.
    attack_range=1, # How far this agent's attack will reach.
    attack_strength=1.0, # How powerful the agent's attack is.
    attack_accuracy=1.0, # Probability of successful attack
    initial_position=None # Episode-starting position. If None, then random within the region.
) for i in range(5)}

# # Hunters try to eat all the foraging agents before all the food disappears.
hunters = {f'hunter{i}': HuntingForagingAgent(
    id=f'hunter{i}',
    agent_view=2, # Partial Observation Mask: how far away this agent can see other agents.
    team=3, # Which team this agent is on
    move_range=1, # How far the agent can move within a single step.
    min_health=0.0, # If the agent's health falls below this value, it will die.
    max_health=1.0, # Agent's health cannot grow above this value.
    attack_range=1, # How far this agent's attack will reach.
    attack_strength=1.0, # How powerful the agent's attack is.
    attack_accuracy=1.0, # Probability of successful attack
    initial_position=None # Episode-starting position. If None, then random within the region.
) for i in range(2)}

agents = {**food, **foragers, **hunters}

# Instantiate the environment

# Set the size of the map
region = 20

# Determine which teams can "attack" each other. In this scenario, team 2 is the
# foragers, and they can attack the food, which is team 1. Team 3 is the hunters
# and they can attack the foragers. So we setup a matrix that represents this.
import numpy as np
team_attack_matrix = np.zeros((4, 4))
team_attack_matrix[2, 1] = 1 # Foragers can attack food
team_attack_matrix[3, 2] = 1 # Hunters can attack foragers
env = HuntingForagingEnv(
    region=region, # The size of the region, both x and y
    number_of_teams=3, # The number of teams
    agents=agents, # Give the environment the dictionary of agents we created above
    team_attack_matrix=team_attack_matrix,
    # attack_norm=np.inf, # The norm to use. Default is np.inf, which means that
    #   the attack radius is square box around the agent
)

# --- Prepare the environment for use with RLlib --- #

# Now that you've created the environment, you must wrap it with a simulation manager,
# which controls the timing of the simulation step.
from abmarl.managers import AllStepManager # All agents take the step at the same time
env = AllStepManager(env)

# We must wrap the environment with the MultiAgentWrapper so that it
# works with RLlib
from abmarl.external.rllib_multiagentenv_wrapper import MultiAgentWrapper
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
    if agents[agent_id].team == 2:
        return 'foragers'
    else:
        return 'hunters'

# USE FOR DEBUGGING
# print(agents['forager0'].action_space)
# print(agents['forager0'].observation_space)
# print(agents['hunter0'].action_space)
# print(agents['hunter0'].observation_space)
# # for agent in agents:
# #     print(policy_mapping_fn(agent))
# import sys; sys.exit()


# --------------------------- #
# --- Setup the algorithm --- #
# --------------------------- #

# Full list of supported algorithms here:
# https://docs.ray.io/en/releases-0.8.5/rllib-algorithms.html
algo_name = 'A2C'


# ------------------ #
# --- Parameters --- #
# ------------------ #

# List of common ray_tune parameters here:
# https://docs.ray.io/en/latest/rllib-training.html#common-parameters
params = {
    'experiment': {
        'title': '{}'.format('ManyForager_5-ManySmartPredator_2-GridTeamObs-View_3-PenalizeDeath'),
        'sim_creator': lambda config=None: env,
    },
    'ray_tune': {
        'run_or_experiment': algo_name,
        'checkpoint_freq': 10,
        'checkpoint_at_end': True,
        'stop': {
            'episodes_total': 200,
        },
        'verbose': 2,
        'config': {
            # --- Environment ---
            'env': "HuntingForaging",
            'horizon': 200,
            'env_config': {},
            # --- Multiagent ---
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_mapping_fn,
            },
            "num_workers": 0,
            "num_envs_per_worker": 1, # This must be 1 because we are not "threadsafe"
            "rollout_fragment_length": 200,
            "batch_mode": "complete_episodes",
            "train_batch_size": 1000,
        },
    }
}


# ---------------------------- #
# --- Random demonstration --- #
# ---------------------------- #

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    shape_dict = {
        1: 's',
        2: 'o',
        3: 'd'
    }

    obs = env.reset()
    import pprint
    pprint.pprint(obs)
    env.render(fig=fig, shape_dict=shape_dict)

    for _ in range(100):
        action_dict = {
            agent.id: agent.action_space.sample()
            for agent in agents.values()
            if agent.is_alive and isinstance(agent, HuntingForagingAgent)
        }
        obs, _, done, _ = env.step(action_dict)
        env.render(fig=fig, shape_dict=shape_dict)
        if done['__all__']:
            break
        # if action_dict['forager0'] == 9:
        #     print('Attack occured!')
            # print(obs['forager0']['life'])
            # plt.pause(1)
