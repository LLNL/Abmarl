
from abc import ABC, abstractmethod

from abmarl.sim.gridworld.base import GridWorldBaseComponent


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


class OneTeamRemainingDone(ActiveDone):
    """
    Inactive agents are indicated as done.

    If the only active agents are those who are all of the same encoding, then
    the simulation ends.
    """
    def get_all_done(self, **kwargs):
        """
        Return true if all active agents have the same encoding. Otherwise,
        return false.
        """
        encodings = set(agent.encoding for agent in self.agents.values() if agent.active)
        return len(encodings) <= 1
