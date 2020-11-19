
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import GridWorldTeamsEnv, GridWorldTeamAgent
from admiral.component_envs.movement import GridMovementEnv, GridMovementAgent
from admiral.component_envs.attacking import GridAttackingTeamEnv, GridAttackingTeamAgent

class PreyAgent(GridWorldTeamAgent, GridMovementAgent):
    pass

class PredatorAgent(GridMovementAgent, GridAttackingTeamAgent):
    pass
# TODO: Resolve the Method resolution bug between world and attacking agents

class PredatorPreyEnv:
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.world = GridWorldTeamsEnv(**kwargs)
        self.movement = GridMovementEnv(**kwargs)
        self.attacking = GridAttackingTeamEnv(**kwargs)

        self.attacking_record = []
    
    def reset(self, **kwargs):
        self.world.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if 'move' in action:
                agent.position = self.movement.process_move(agent.position, action['move'])
            if action.get('attack', False):
                attacked_agent = self.attacking.process_attack(agent)
                if attacked_agent is not None:
                #     self.agents[attacked_agent].health -= agent.attack_strength
                #     agent.health += agent.attack_strength # Gain health from a good attack.
                    self.attacking_record.append(agent.id + " attacked " + attacked_agent)
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        # render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}
        # self.world.render(fig=fig, render_condition=render_condition, shape_dict=shape, **kwargs)
        self.world.render(fig=fig, shape_dict=shape, **kwargs)
        for record in self.attacking_record:
            print(record)
        self.attacking_record.clear()
        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        return {'agents': self.world.get_obs(agent_id)}


prey = {f'prey{i}': PreyAgent(id=f'prey{i}', view=5, team=1, move=1) for i in range(7)}
predators = {f'predator{i}': PredatorAgent(id=f'predator{i}', view=2, team=2, move=1, attack_range=1, attack_strength=0.24) for i in range(2)}
agents = {**prey, **predators}
region = 10
env = PredatorPreyEnv(
    region=region,
    agents=agents,
    number_of_teams=2
)
env.reset()
print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
fig = plt.gcf()
env.render(fig=fig)

for _ in range(100):
    action_dict = {}
    for agent_id, agent in env.agents.items():
        if isinstance(agent, PreyAgent):
            action_dict[agent_id] = {
                'move': agent.action_space['move'].sample(),
            }
        elif isinstance(agent, PredatorAgent):
            action_dict[agent_id] = {
                'move': agent.action_space['move'].sample(),
                'attack': agent.action_space['attack'].sample(),
            }
    env.step(action_dict)
    env.render(fig=fig)
    x = []

