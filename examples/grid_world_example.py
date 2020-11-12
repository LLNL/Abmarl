
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.world import WorldAgent
from admiral.component_envs.movement import GridMovementEnv
from admiral.component_envs.resources import GridResourceEnv

class ResourceManagementEnv:
    def __init__(self, world=None, movement=None, resource=None):
        # TODO: require splitting world and movement
        self.world = world
        self.movement = movement
        self.resource = resource
    
    def reset(self, **kwargs):
        self.resource.reset(**kwargs)
        self.movement.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.world.agents[agent_id]
            if 'move' in action:
                agent.position = self.movement.process_move(agent.position, action['move'])
                # TODO: Requires an update on process_move function
            if 'harvest' in action:
                amount_harvested = self.resource.process_harvest(agent.position, action['harvest'])
                # Do something with the amount that is harvested.
    
    def render(self, **kwargs):
        self.resource.render(**kwargs)
        self.world.render(**kwargs)

env = GridMovementEnv(
    region=10,
    agents={f'agent{i}': WorldAgent(id=f'agent{i}') for i in range(4)}
)
env.reset()
fig = plt.gcf()
env.render(fig=fig)

for _ in range(20):
    fig.clear()
    for agent in env.agents.values():
        env.process_move(agent, np.random.randint(-1, 2, size=(2,)))
    env.render(fig=fig)
    plt.plot()
    plt.pause(1e-17)
