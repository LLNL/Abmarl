
import numpy as np

from admiral.envs import Agent
from admiral.component_envs.world import GridWorldAgent
from admiral.component_envs.component import Component

class GridMovementAgent(Agent):
    def __init__(self, move=None, **kwargs):
        assert move is not None, "move must be an integer"
        self.move = move
        super().__init__(**kwargs)
    
    @property
    def configured(self):
        return super().configured and self.move is not None

class GridMovementComponent(Component):
    """
    Agents in the GridWorld can move around.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer"
        self.region = region
        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, GridMovementAgent):
                agent.action_space['move'] = Box(-agent.move, agent.move, (2,), np.int)

    def act(self, agent, move, **kwargs):
        new_position = agent.position + move
        if 0 <= new_position[0] < self.region and \
            0 <= new_position[1] < self.region: # Still inside the boundary, good move
            agent.position = new_position
