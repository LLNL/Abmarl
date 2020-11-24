
import numpy as np

from admiral.component_envs.world import GridWorldAgent
from admiral.component_envs.component import Component

# TODO: Consider process_move will take the agent as input argument and output
# the new position. This would mean that GridMovementAgents will inherit from
# GridWorldAgent. The step function will then update and check that the agent is
# of the right type before processing the move action? Right now, the environment
# has to check if move is in the action.
class GridWorldMovementAgent(GridWorldAgent):
    def __init__(self, move=None, **kwargs):
        assert move is not None, "move must be an integer"
        self.move = move
        super().__init__(**kwargs)
    
    @property
    def configured(self):
        return super().configured and self.move is not None

class GridWorldMovementComponent(Component):
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
            if isinstance(agent, GridWorldMovementAgent):
                agent.action_space['move'] = Box(-agent.move, agent.move, (2,), np.int)

    def act(self, agent, move, **kwargs):
        if isinstance(agent, GridWorldMovementAgent):
            new_position = agent.position + move
            if 0 <= new_position[0] < self.region and \
               0 <= new_position[1] < self.region: # Still inside the boundary, good move
                agent.position = new_position
