
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent

class MovementEnv(ABC):
    """
    MovementEnv processes agent movement in the world in which they live. Provides
    the process_move api.
    """
    @abstractmethod
    def process_move(self, agent, move, **kwargs):
        """
        Move the agent some amount.
        """
        pass

class GridMovementEnv(MovementEnv):
    """
    Agents in the GridWorld can move around.
    """
    def process_move(self, position, move, **kwargs):
        """
        Move the agent according to the move action. Returns the propsed new position.
        """
        return position + move
