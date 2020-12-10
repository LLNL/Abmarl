
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.position import PositionAgent

class MovementAgent(Agent):
    """
    Agents can move in the grid up to some number of spaces away.

    move (int):
        The maximum number of cells away that the agent can move.
    """
    def __init__(self, move_range=None, **kwargs):
        super().__init__(**kwargs)
        assert move_range is not None, "move_range must be an integer"
        self.move_range = move_range
    
    @property
    def configured(self):
        """
        Agents are configured if the move parameter is set.
        """
        return super().configured and self.move_range is not None

class GridMovementActor:
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = None
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, MovementAgent):
                agent.action_space['move'] = Box(-agent.move, agent.move, (2,), np.int)

    def process_move(self, agent_id, move, **kwargs):
        self.position.modify_position(agent_id, move, **kwargs)
