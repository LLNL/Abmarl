
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent

class MovementEnv(ABC):
    """
    MovementEnv processes agent movement in the world in which they live. Agent
    movement is bounded by the size of the region.
    
    Provides the process_move api.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region

    @abstractmethod
    def process_move(self, position, move, **kwargs):
        """
        Update the position by the movement action.
        """
        pass

class GridMovementAgent(Agent):
    def __init__(self, move=None, **kwargs):
        super().__init__(**kwargs)
        self.move = move
    
    @property
    def configured(self):
        return super().configured and self.move

class GridMovementEnv(MovementEnv):
    """
    Agents in the GridWorld can move around.
    """
    def __init__(self, agents=None, **kwargs):
        super().__init__(**kwargs)
        assert type(agents) is dict, "agents must be a dict"
        # Append action space where relevant
        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, GridMovementAgent):
                agent.action_space['move'] = Box(-agent.move, agent.move, (2,), np.int)
        self.agents = agents

    def process_move(self, position, move, **kwargs):
        """
        Move the agent according to the move action. Returns the proposed new position.
        """
        new_position = position + move
        if 0 <= new_position[0] < self.region and \
           0 <= new_position[1] < self.region: # Still inside the boundary, good move
            return new_position
        else:
            return position
