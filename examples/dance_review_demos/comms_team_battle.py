
# ----------------------------- #
# --- Setup the environment --- #
# ----------------------------- #

# --- Create the agents and the environment --- #

# Import the simulation environment and agents
from admiral.envs.components.examples.comms_team_battle import CommunicatingAgent, BattleAgent, TeamBattleCommsEnv

number_of_teams = 2
region = 15

# Instatiate the agents that will operate in this environment. All possible agent
# attributes are listed below
import numpy as np
agents = {
    'agent0': CommunicatingAgent(id='agent0', initial_position=np.array([7, 7]), team=1, broadcast_range=region, agent_view=region),
    'agent1': BattleAgent(id='agent1', initial_position=np.array([0, 4]), team=1, agent_view=0, attack_range=1, move_range=1, attack_strength=1),
    'agent2': BattleAgent(id='agent2', initial_position=np.array([0, 7]), team=1, agent_view=0, attack_range=1, move_range=1, attack_strength=1),
    'agent3': BattleAgent(id='agent3', initial_position=np.array([0, 10]), team=1, agent_view=0, attack_range=1, move_range=1, attack_strength=1),
    'agent4': BattleAgent(id='agent4', initial_position=np.array([14, 4]), team=2, agent_view=0, attack_range=1, move_range=1, attack_strength=1),
    'agent5': BattleAgent(id='agent5', initial_position=np.array([14, 7]), team=2, agent_view=0, attack_range=1, move_range=1, attack_strength=1),
    'agent6': BattleAgent(id='agent6', initial_position=np.array([14, 10]), team=2, agent_view=0, attack_range=1, move_range=1, attack_strength=1),
}

# Instantiate the environment
env = TeamBattleCommsEnv(
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
env_name = "CommsTeamBattle"
register_env(env_name, lambda env_config: env)


# -------------------------- #
# --- Setup the policies --- #
# -------------------------- #

# Here we have it setup so that every agent on a team trains the same policy.
# Because every agent has the same observation and action space, we can just use
# the specs from one of the agent to define the policies' inputs and outputs.
policies = {
    'comms_agent': (None, agents['agent0'].observation_space, agents['agent0'].action_space, {}),
    'battle_team_1': (None, agents['agent1'].observation_space, agents['agent1'].action_space, {}),
    'battle_team_2': (None, agents['agent4'].observation_space, agents['agent4'].action_space, {})
}
def policy_mapping_fn(agent_id):
    if isinstance(agents[agent_id], CommunicatingAgent):
        return 'comms_agent'
    elif agents[agent_id].team == 1:
        return 'battle_team_1'
    else:
        return 'battle_team_2'

# USE FOR DEBUGGING
# print(agents['agent4'].action_space)
# print(agents['agent4'].observation_space)
# # for agent in agents:
# #     print(policy_mapping_fn(agent))
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
        'title': '{}'.format('CommsTeamBattle'),
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
            'env': env_name,
            'horizon': 200,
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


# ---------------------------- #
# --- Random demonstration --- #
# ---------------------------- #

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    env.reset()
    shape_dict={1: 's', 2:'o'}
    env.render(fig=fig, shape_dict=shape_dict)

    for _ in range(100):
        action_dict = {agent.id: agent.action_space.sample() for agent in agents.values() if agent.is_alive}
        _, _, done, _ = env.step(action_dict)
        env.render(fig=fig, shape_dict=shape_dict)
        if done['__all__']:
            break