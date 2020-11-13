
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import GridWorldEnv
from admiral.component_envs.movement import GridMovementEnv
from admiral.component_envs.resources import GridResourceEnv
from admiral.component_envs.attacking import GridAttackingEnv, AttackingTeamAgent

class FightForResourcesEnv:
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.world = GridWorldEnv(**kwargs)
        self.resource = GridResourceEnv(**kwargs)
        self.movement = GridMovementEnv(**kwargs)
        self.attacking = GridAttackingEnv(**kwargs)

        self.attacking_record = []
    
    def reset(self, **kwargs):
        self.world.reset(**kwargs)
        self.resource.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if 'attack' in action:
                attacked_agent = self.attacking.process_attack(agent)
                if attacked_agent is not None:
                    self.attacking_record.append(agent.id + " attacked " + attacked_agent)
            if 'move' in action:
                agent.position = self.movement.process_move(agent.position, action['move'])
            if 'harvest' in action:
                amount_harvested = self.resource.process_harvest(tuple(agent.position), action['harvest'])
        
        self.resource.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        self.resource.render(fig=fig, **kwargs)
        self.world.render(fig=fig, **kwargs)
        for record in self.attacking_record:
            print(record)
        self.attacking_record.clear()
        plt.plot()
        plt.pause(1e-6)

env = FightForResourcesEnv(
    region=10,
    agents={f'agent{i}': AttackingTeamAgent(
        id=f'agent{i}',
        attack_range=1,
        team=i%2
    ) for i in range(6)}
)
env.reset()
fig = plt.gcf()
env.render(fig=fig)

for _ in range(20):
    action_dict = {agent_id: {} for agent_id in env.agents}
    for agent_id, agent in env.agents.items():
        action_dict[agent_id] = {
            'move': np.random.randint(-1, 2, size=(2,)),
            'harvest': np.random.uniform(0, 1),
            'attack': np.random.randint(0, 2)
        }
    env.step(action_dict)
    env.render(fig=fig)
    x = []
