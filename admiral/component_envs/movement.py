
import numpy as np

from admiral.envs import Agent

# TODO: Consider process_move will take the agent as input argument and output
# the new position. This would mean that GridMovementAgents will inherit from
# GridWorldAgent. The step function will then update and check that the agent is
# of the right type before processing the move action? Right now, the environment
# has to check if move is in the action.

def GridMovementAgent(move=None, **kwargs):
    return {
        **Agent(**kwargs),
        'move': move,
    }

class GridMovementEnv:
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
            if 'move' in agent:
                agent.action_space['move'] = Box(-agent.move, agent.move, (2,), np.int)

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
