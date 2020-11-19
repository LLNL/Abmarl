
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import GridWorldTeamsEnv, GridWorldTeamAgent

prey = {f'prey{i}': GridWorldTeamAgent(id=f'prey{i}', view=5, team=1) for i in range(7)}
predators = {f'predator{i}': GridWorldTeamAgent(id=f'predator{i}', view=2, team=2) for i in range(2)}
agents = {**prey, **predators}
region = 10
env = GridWorldTeamsEnv(
    region=region,
    agents=agents,
    number_of_teams=2
)
env.reset()
print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
fig = plt.gcf()

# render_condition = {agent.id: agent.is_alive for agent in env.agents.values()}
shape = {agent.id: 'o' if agent.team == 1 else 's' for agent in env.agents.values()}
env.render(fig=fig, shape_dict=shape)

plt.show()
