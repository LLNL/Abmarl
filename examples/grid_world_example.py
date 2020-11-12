
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import WorldAgent, GridWorldEnv
from admiral.component_envs.movement import GridMovementEnv
from admiral.component_envs.resources import GridResourceEnv

class ResourceManagementEnv:
    def __init__(self, world=None, movement=None, resource=None):
        self.world = world
        self.agents = self.world.agents
        self.movement = movement
        self.resource = resource
    
    def reset(self, **kwargs):
        self.world.reset(**kwargs)
        self.resource.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if 'move' in action:
                new_position = self.movement.process_move(agent.position, action['move'])
                if 0 <= new_position[0] < self.world.region and \
                    0 <= new_position[1] < self.world.region: # Still inside the boundary, good move
                    agent.position = new_position
            if 'harvest' in action:
                amount_harvested = self.resource.process_harvest(tuple(agent.position), action['harvest'])
        
        self.resource.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        self.resource.render(fig=fig, **kwargs)
        self.world.render(fig=fig, **kwargs)
        plt.plot()
        plt.pause(1e-6)

env = ResourceManagementEnv(
    world=GridWorldEnv(
        region=10,
        agents={f'agent{i}': WorldAgent(id=f'agent{i}') for i in range(4)}
    ),
    movement=GridMovementEnv(),
    resource=GridResourceEnv(region=10)
)
env.reset()
fig = plt.gcf()
env.render(fig=fig)

for _ in range(20):
    action_dict = {agent_id: {} for agent_id in env.agents}
    for agent_id, agent in env.agents.items():
        action_dict[agent_id] = {
            'move': np.random.randint(-1, 2, size=(2,)),
            'harvest': np.random.uniform(0, 1)
        }
        env.step(action_dict)
    env.render(fig=fig)
