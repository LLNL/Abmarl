
from abc import ABC, abstractmethod

from abmarl.sim.grid_world import GridWorldBaseComponent, MovingAgent

class ActionBaseComponent(GridWorldBaseComponent, ABC):
    """
    Abstract ActionComponent class from which all ActionComponents will inherit.
    """
    @abstractmethod
    def process_action(self, agent, action, **kwargs):
        """
        Process the agent's action.

        Args:
            agent: The acting agent.
            action: The relevant action.
        """
        pass

    @property
    @abstractmethod
    def key(self):
        """
        The key in the action dictionary.
        """
        pass

    # @property
    # @abstractmethod
    # def corresponding_agent(self):
    #     """
    #     The agent class that works with this Action component.
    #     """
    #     pass


class MoveAction(ActionBaseComponent):
    """
    Process moving agents.
    """    
    def process_action(self, agent, action_dict, **kwargs):
        action = action_dict[self.key]
        new_position = agent.position + action
        if 0 <= new_position[0] < self.rows and \
                0 <= new_position[1] < self.cols and \
                self.grid[new_position[0], new_position[1]] is None:
            self.grid[agent.position[0], agent.position[1]] = None
            agent.position = new_position
            self.grid[agent.position[0], agent.position[1]] = agent

    @property
    def key(self):
        return "move"
    
    # @property
    # def corresponding_agent(self):
    #     return MovingAgent
