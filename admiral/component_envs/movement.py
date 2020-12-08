
import numpy as np

from admiral.envs import Agent
from admiral.component_envs.position import GridPositionAgent

class GridMovementAgent(Agent):
    """
    Agents can move in the grid up to some number of spaces away.

    move (int):
        The maximum number of cells away that the agent can move.
    """
    def __init__(self, move=None, **kwargs):
        super().__init__(**kwargs)
        assert move is not None, "move must be an integer"
        self.move = move
    
    @property
    def configured(self):
        """
        Agents are configured if the move parameter is set.
        """
        return super().configured and self.move is not None

class GridMovementComponent:
    """
    Agents can move around in the grid. This component processes the move action.
    If the new position is out of bounds, then the position is not updated.

    The agents' action space is appended with Box(-agent.move, agent.move, (2,), np.int),
    indicating that the agent can move in 2 dimensions up to its maximum distance.

    region (int):
        The size of the region. This is needed to determine if the agent has attempted
        to move out of bounds or not.

    agents (dict):
        The dictionary of agents. Because the movement is based on the agent's
        posiiton, the agent must be GridPositionAgents.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer"
        self.region = region
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, GridPositionAgent)
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, GridMovementAgent):
                agent.action_space['move'] = Box(-agent.move, agent.move, (2,), np.int)

    def process_move(self, agent, move, **kwargs):
        """
        Process the agent's new position. If the agent attempts to move outside
        the region, the movement does not happen.
        """
        new_position = agent.position + move
        if 0 <= new_position[0] < self.region and \
            0 <= new_position[1] < self.region: # Still inside the boundary, good move
            agent.position = new_position
