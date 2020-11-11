
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent
from admiral.component_envs.world import WorldEnv, GridWorldEnv

class MovementEnv(WorldEnv):
    """
    MovementEnv processes agent movement in the WorldEnv in which they live. Provides
    the process_move api.
    """
    @abstractmethod
    def process_move(self, agent, move, **kwargs):
        """
        Move the agent some amount.
        """
        pass

class GridMovementEnv(MovementEnv, GridWorldEnv):
    """
    Agents in the GridWorld can move around.
    """
    def process_move(self, agent, move, **kwargs):
        """
        Move the agent according to the move action. Returns true of the move is
        successful. Returns false otherwise (e.g. if the agent attempts to move
        outside of the region).
        """
        if 0 <= agent.position[0] + move[0] < self.region and \
           0 <= agent.position[1] + move[1] < self.region: # Still inside the boundary, good move
            agent.position += move 
            return True
        else:
            return False
