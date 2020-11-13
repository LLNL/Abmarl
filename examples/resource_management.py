
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import WorldAgent, GridWorldEnv
from admiral.component_envs.movement import GridMovementEnv
from admiral.component_envs.resources import GridResourceEnv
from admiral.component_envs.death_life import DyingAgent, DyingEnv

class ResourceManagementAgent(WorldAgent, DyingAgent):
    pass

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
        fig.clear(DyingAgent)
        self.resource.render(fig=fig, **kwargs)
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        self.world.render(fig=fig, render_condition=render_condition, **kwargs)
        plt.plot()
        plt.pause(1e-6)
        self.dying.render()

env = ResourceManagementEnv(
    region=10,
    agents={f'agent{i}': ResourceManagementAgent(id=f'agent{i}') for i in range(4)}
)
env.reset()
fig = plt.gcf()
env.render(fig=fig)

for _ in range(50):
    action_dict = {agent_id: {} for agent_id in env.agents}
    for agent_id, agent in env.agents.items():
        action_dict[agent_id] = {
            'move': np.random.randint(-1, 2, size=(2,)),
            'harvest': np.random.uniform(0, 1)
        }
    env.step(action_dict)
    env.render(fig=fig)
