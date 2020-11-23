
from box import Box as DictObj
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import GridWorldEnv, GridWorldObservingAgent
from admiral.component_envs.movement import GridMovementEnv, GridMovementAgent
from admiral.component_envs.resources import GridResourceEnv, GridResourceHarvestingAndObservingAgent
from admiral.component_envs.death_life import DyingAgent, DyingEnv

def ResourceManagementAgent(**kwargs):
    return DictObj({
        **DyingAgent(**kwargs),
        **GridMovementAgent(**kwargs),
        **GridResourceHarvestingAndObservingAgent(**kwargs),
    })

class ResourceManagementEnv:
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.world = GridWorldEnv(**kwargs)
        self.resource = GridResourceEnv(**kwargs)
        self.movement = GridMovementEnv(**kwargs)
        self.dying = DyingEnv(**kwargs)
    
    def reset(self, **kwargs):
        self.world.reset(**kwargs)
        self.resource.reset(**kwargs)
        self.dying.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if agent.is_alive:
                if 'move' in action:
                    agent.position = self.movement.process_move(agent.position, action['move'])
                if 'harvest' in action:
                    amount_harvested = self.resource.process_harvest(tuple(agent.position), action['harvest'])
                    agent.health += amount_harvested
                self.dying.apply_entropy(agent)
                self.dying.process_death(agent)
        self.resource.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        self.resource.render(fig=fig, **kwargs)
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        self.world.render(fig=fig, render_condition=render_condition, **kwargs)
        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        return {'agents': self.world.get_obs(agent_id), 'resources': self.resource.get_obs(agent_id)}

agents = {f'agent{i}': ResourceManagementAgent(id=f'agent{i}', view=2, move=1, max_harvest=1.0) for i in range(4)}
env = ResourceManagementEnv(
    region=10,
    agents=agents
)
env.reset()
print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
fig = plt.gcf()
env.render(fig=fig)

for _ in range(50):
    action_dict = {agent_id: {} for agent_id in env.agents}
    for agent_id, agent in env.agents.items():
        action_dict[agent_id] = {
            'move': agent.action_space['move'].sample(),
            'harvest': agent.action_space['harvest'].sample()
        }
    env.step(action_dict)
    env.render(fig=fig)
