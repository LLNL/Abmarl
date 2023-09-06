
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent
from abmarl.sim.gridworld.agent import GridWorldAgent


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


class TargetAgentDone(DoneBaseComponent):
    """
    Agents are done when they overlap their target.

    The target is prescribed per agent.
    """
    def __init__(self, target_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.target_mapping = target_mapping

    @property
    def target_mapping(self):
        """
        Maps the agent to its respective target.

        Mapping is done via the agents' ids.
        """
        return self._target_mapping

    @target_mapping.setter
    def target_mapping(self, value):
        assert type(value) is dict, "Target mapping must be a dictionary."
        for agent_id, target_id in value.items():
            assert agent_id in self.agents, f"{agent_id} must be an agent in the simulation."
            assert isinstance(self.agents[agent_id], GridWorldAgent), \
                f"{agent_id} must be a GridWorldAgent."
            assert target_id in self.agents, "Target must be an agent in the simulation."
            assert isinstance(self.agents[target_id], GridWorldAgent), \
                "Target must be a GridWorldAgent."
        self._target_mapping = value

    def get_done(self, agent, **kwarg):
        return np.array_equal(
            agent.position,
            self.agents[self.target_mapping[agent.id]].position
        )

    def get_all_done(self, **kwargs):
        return all([
            self.get_done(self.agents[agent_id]) for agent_id in self.target_mapping
        ])


class TargetDestroyedDone(DoneBaseComponent):
    """
    Agents are done when their target agent becomes inactive.
    """
    def __init__(self, target_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.target_mapping = target_mapping

    @property
    def target_mapping(self):
        """
        Maps the agent to its respective target.

        Mapping is done via the agents' ids.
        """
        return self._target_mapping

    @target_mapping.setter
    def target_mapping(self, value):
        assert type(value) is dict, "Target mapping must be a dictionary."
        for agent_id, target_id in value.items():
            assert agent_id in self.agents, f"{agent_id} must be an agent in the simulation."
            assert isinstance(self.agents[agent_id], GridWorldAgent), \
                f"{agent_id} must be a GridWorldAgent."
            assert target_id in self.agents, "Target must be an agent in the simulation."
            assert isinstance(self.agents[target_id], GridWorldAgent), \
                "Target must be a GridWorldAgent."
        self._target_mapping = value

    def get_done(self, agent, **kwarg):
        return not self.agents[self.target_mapping[agent.id]].active

    def get_all_done(self, **kwargs):
        return all([
            self.get_done(self.agents[agent_id]) for agent_id in self.target_mapping
        ])


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
