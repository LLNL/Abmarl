
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.agent import HealthAgent, TeamAgent

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


class DeadDone(DoneBaseComponent):
    """
    Dead agents are indicated as done. Additionally, the simulation is over when
    all the agents are dead.
    """

    def get_done(self, agent, **kwargs):
        """
        Return True if the agent is dead. Otherwise, return False.
        """
        if isinstance(agent, HealthAgent):
            return not agent.is_alive

    def get_all_done(self, **kwargs):
        """
        Return True if all agents are dead. Otherwise, return False.
        """
        for agent in self.agents.values():
            if isinstance(agent, HealthAgent):
                if agent.is_alive:
                    return False
        return True


class TeamDeadDone:
    """
    Dead agents are indicated as done. Additionally, the simulation is over when
    the only agents remaining are on the same team.
    """
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        super().__init__(**kwargs)
        self.number_of_teams = number_of_teams

    @property
    def number_of_teams(self):
        """
        The number of teams in the simulation.

        The number of teams must be a nonnegative integer.
        """
        return self._number_of_teams

    @number_of_teams.setter
    def number_of_teams(self, value):
        assert type(value) is int and 0 <= value, "Number of teams must be a nonnegative integer."
        self._number_of_teams = value

    def get_done(self, agent, **kwargs):
        """
        Returns:
            True if the agent is dead. Otherwise, return False.
        """
        if isinstance(agent, HealthAgent):
            return not agent.is_alive

    def get_all_done(self, **kwargs):
        """
        Returns:
            True if the only agents left alive are all on the same team. Otherwise,
            return false.
        """
        team = np.zeros(self.number_of_teams)
        for agent in self.agents.values():
            if isinstance(agent, HealthAgent) and isinstance(agent, TeamAgent) and agent.is_alive:
                team[agent.team] += 1
        return sum(team != 0) <= 1
