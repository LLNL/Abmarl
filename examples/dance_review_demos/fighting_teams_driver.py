
# ----------------------------- #
# --- Setup the environment --- #
# ----------------------------- #

# Import the simulation environment and agents
from admiral.envs.components.examples.fighting_teams import FightingTeamsEnv, FightingTeamAgent

number_of_teams = 3

# Instatiate the agents that will operate in this environment. All possible agent
# attributes are listed below
agents = {f'agent{i}': FightingTeamAgent(
    id=f'agent{i}', # Every agent needs a unique id.
    attack_range=1, # How far this agent's attack will reach.
    attack_strength=1.0, # How powerful the agent's attack is.
    team=i%number_of_teams, # Every agent is on a team
    move_range=1, # How far the agent can move within a single step.
    min_health=0.0, # If the agent's health falls below this value, it will die.
    max_health=1.0, # Agent's health cannot grow above this value.
    initial_health=None, # Episode-starting health. If None, then random between min and max health. 
    initial_position=None # Episode-starting position. If None, then random within the region.
) for i in range(24)}

# Instantiate the environment
env = FightingTeamsEnv(
    number_of_teams=number_of_teams, # Environment must be told the number of teams
    region=10, # Size of the region, in both x and y
    # attack_norm=np.inf, # The norm to use. Default is np.inf, which means that the attack radius is square box around the agent
    agents=agents # Give the environment the dictionary of agents we created above
)

# from admiral.envs.wrappers import MultiAgentWrapper
# from ray.tune.registry import register_env
# env_name = "EnvironmentName"
# register_env(env_name, lambda env_config: MultiAgentWrapper.wrap(env))


# # -------------------------- #
# # --- Setup the policies --- #
# # -------------------------- #

# from admiral.pols import HeuristicPolicy

# class CustomHeuristicPolicy(HeuristicPolicy):
#     """A custom heuristic policy for you the design"""
#     def compute_actions(self, obs_batch, *args, **kwargs):
#         return [some_action for _ in obs_batch], [], {}

# policies = {
#     'policy_0_name': (None, agents[0].observation_space, agents[0].action_space, {}),
#     'policy_1_name': (None, agents[1].observation_space, agents[1].action_space, {}),
#     'policy_2_name': (None, agents[2].observation_space, agents[2].action_space, {})
# }
# def policy_mapping_fn(agent_id):
#     pass # Map the agent id to the policy you want that agent to train.


# # --------------------------- #
# # --- Setup the algorithm --- #
# # --------------------------- #

# # Full list of supported algorithms here: https://docs.ray.io/en/releases-0.8.5/rllib-algorithms.html
# algo_name = 'PG'


# # ------------------ #
# # --- Parameters --- #
# # ------------------ #

# # List of common ray_tune parameters here: https://docs.ray.io/en/latest/rllib-training.html#common-parameters
# params = {
#     'experiment': {
#         'title': '{}'.format('The-title-of-this-experiment'),
#     },
#     'ray_tune': {
#         'run_or_experiment': algo_name,
#         'stop': {
#             # Stopping criteria
#         },
#         'config': {
#             # --- Environment ---
#             'env': env_name,
#             'env_config': env_config,
#             # --- Multiagent ---
#             'multiagent': {
#                 'policies': policies,
#                 'policy_mapping_fn': policy_mapping_fn,
#             },
#         },
#     }
# }