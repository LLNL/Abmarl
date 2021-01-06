
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.agent import GridMovementAgent
from admiral.envs.components.agent import PositionAgent


class GridMovementActor:
    """
    Provides the necessary action space for agents who can move and processes such
    movements.

    position (PositionState):
        The position state handler. Needed to modify the agents' positions.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, GridMovementAgent):
                agent.action_space['move'] = Box(-agent.move_range, agent.move_range, (2,), np.int)

    def process_move(self, moving_agent, move, **kwargs):
        """
        Determine the agent's new position based on its move action.

        moving_agent (GridMovementAgent):
            The agent that moves.
        
        move (np.array):
            How much the agent would like to move in row and column.
        
        return (np.array):
            How much the agent has moved in row and column. This can be different
            from the desired move if the position update was invalid.
        """
        if isinstance(moving_agent, GridMovementAgent) and isinstance(moving_agent, PositionAgent):
            position_before = moving_agent.position
            self.position.modify_position(moving_agent, move, **kwargs)
            return position_before - moving_agent.position
