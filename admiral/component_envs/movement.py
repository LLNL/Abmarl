
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent
from admiral.component_envs import WorldAgent

class MovementEnv(ABC):
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        self.agents = agents if agents is not None else {}
        for agent in self.agents.values():
            assert isinstance(agent, WorldAgent)

    @abstractmethod
    def _process_move(self, agent, move, **kwargs):
        """
        Move the agent some amount.
        """
        pass

class GridMovementEnv(MovementEnv):
    def _process_move(self, agent, move, **kwargs):
        if 0 <= agent.position[0] + move[0] < self.region and \
           0 <= agent.position[1] + move[1] < self.region: # Still inside the boundary, good move
            agent.position += move 
            return True
        else:
            return False
