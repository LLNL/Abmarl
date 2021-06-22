
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.agent import HealthAgent

class DoneBaseComponent(GridWorldBaseComponent, ABC):
    """
    Abstract Done Component class from which all Done Components will inherit.
    """
    @abstractmethod
    def get_done(self, agent, **kwargs):
        """
        Determine if an agent is done in this step.

        Args:
            agent: The agent we are querying.
        
        Returns:
            True if the agent is done, otherwise False.
        """
        pass

    @abstractmethod
    def get_all_done(self, **kwargs):
        """
        Determine if all the agents are done and/or if the simulation is done.

        Returns:
            True if all agents are done or if the simulation is done. Otherwise
            False.
        """
        pass


class ActiveDone(DoneBaseComponent):
    """
    Inactive agents are indicated as done.
    """
    def get_done(self, agent, **kwargs):
        """
        Return True if the agent is inactive. Otherwise, return False.
        """
        return not agent.active

    def get_all_done(self, **kwargs):
        """
        Return True if all agents are inactive. Otherwise, return False.
        """
        for agent in self.agents.values():
            if agent.active:
                return False
        return True