
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import GridWorldTeamsEnv, GridWorldObservingTeamAgent
from admiral.component_envs.movement import GridMovementEnv, GridMovementAgent
from admiral.component_envs.attacking import GridAttackingTeamEnv, GridWorldAttackingTeamAgent
from admiral.component_envs.death_life import DyingAgent, DyingEnv


# TODO: Figure out a better way than multiple inheritance to share parameters,
# becuase this is not working well for us. All we're really doing is giving the
# agents attributes, which is like giving them key, value pairs in a dictionary.
# Perhaps each class can just be a dictionary, and inheritance can be handled by
# adding to that dictionary?
# class FightingTeamsAgent:
#     def __init__(self, min_health=0.0, max_health=1.0, attack_range=None, attack_strength=None,
#             move=None, team=None, view=None):
#         self.min_health = min_health
#         self.max_health = max_health
#         self.health = None
#         self.is_alive = True
#         self.team = team
#         self.attack_range = attack_range
#         self.attack_strength = attack_strength
#         self.move = move



class FightingTeamsAgent(DyingAgent, GridWorldAttackingTeamAgent, GridMovementAgent, GridWorldObservingTeamAgent):
    pass

class FightingTeamsEnv:
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.world = GridWorldTeamsEnv(**kwargs)
        self.movement = GridMovementEnv(**kwargs)
        self.attacking = GridAttackingTeamEnv(**kwargs)
        self.dying = DyingEnv(**kwargs)

        self.attacking_record = []
    
    def reset(self, **kwargs):
        self.world.reset(**kwargs)
        self.dying.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.is_alive:
                if action.get('attack', False):
                    attacked_agent = self.attacking.process_attack(agent)
                    if attacked_agent is not None:
                        self.agents[attacked_agent].health -= agent.attack_strength
                        agent.health += agent.attack_strength # Gain health from a good attack.
                        self.attacking_record.append(agent.id + " attacked " + attacked_agent)
                if 'move' in action:
                    agent.position = self.movement.process_move(agent.position, action['move'])
            
        # Because agents can affect each others' health, we process the dying
        # outside the loop at the end of all the moves. Note: this does not
        # matter in a TurnBasedManager.
        for agent_id in action_dict:
            agent = self.agents[agent_id]
            if agent.is_alive:
                self.dying.process_death(agent)
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}
        self.world.render(fig=fig, render_condition=render_condition, shape_dict=shape, **kwargs)
        for record in self.attacking_record:
            print(record)
        self.attacking_record.clear()
        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        return {'agents': self.world.get_obs(agent_id)}

agents = {f'agent{i}': FightingTeamsAgent(
    id=f'agent{i}', attack_range=1, attack_strength=0.4, team=i%2+1, move=1, view=11
) for i in range(24)}
env = FightingTeamsEnv(
    region=12,
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
        action_dict[agent_id] = {
            'move': agent.action_space['move'].sample(),
            'attack': agent.action_space['attack'].sample(),
        }
    env.step(action_dict)
    env.render(fig=fig)
    x = []
