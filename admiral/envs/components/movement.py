
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.position import PositionAgent

class GridMovementAgent(Agent):
    """
    Agents can move up to some number of spaces away.

    move_range (int):
        The maximum number of cells away that the agent can move.
    """
    def __init__(self, move_range=None, **kwargs):
        super().__init__(**kwargs)
        assert move_range is not None, "move_range must be an integer"
        self.move_range = move_range
    
    @property
    def configured(self):
        """
        Agents are configured if the move_range parameter is set.
        """
        return super().configured and self.move_range is not None

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
        if isinstance(moving_agent, GridMovementAgent) and isinstance(agent, PositionAgent):
            position_before = moving_agent.position
            self.position.modify_position(moving_agent, move, **kwargs)
            return position_before - moving_agent.position
