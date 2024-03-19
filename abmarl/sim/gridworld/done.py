
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


class TargetEncodingInactiveDone(DoneBaseComponent):
    """
    Target encoding mapping indicates which encodings map to each other.

    An agent or simulation is done if all its target encodings are inactive.
    """
    def __init__(self, target_mapping=None, sim_ends_if_one_done=True, **kwargs):
        super().__init__(**kwargs)
        self._encodings_in_sim = {agent.encoding for agent in self.agents.values()}
        self.target_mapping = target_mapping
        self.sim_ends_if_one_done = sim_ends_if_one_done

    @property
    def sim_ends_if_one_done(self):
        """
        Specify if the simulation ends of one team is done.
        """
        return self._sim_ends_if_one_done

    @sim_ends_if_one_done.setter
    def sim_ends_if_one_done(self, value):
        if value is not None:
            assert type(value) is bool, "Value must be True or False."
            self._sim_ends_if_one_done = value
        else:
            self._sim_ends_if_one_done = True

    @property
    def target_mapping(self):
        """
        Maps encodings to their respective target encodings.
        """
        return self._target_mapping

    @target_mapping.setter
    def target_mapping(self, value):
        assert type(value) is dict, "Target mapping must be a dictionary."
        for encoding, target_encoding in value.items():
            assert encoding in self._encodings_in_sim, \
                f"Encoding {encoding} not in the simulation."
            if type(target_encoding) is int: # Target provided as integer
                assert target_encoding in self._encodings_in_sim, \
                    f"Target encoding {target_encoding} not in the simulation."
                assert target_encoding != encoding, "Agent cannot target its own encodings."
                value[encoding] = {target_encoding} # Upgrade encoding to a set for ease
            elif type(target_encoding) is set: # Target provided as a set
                for te in target_encoding:
                    assert te in self._encodings_in_sim, \
                        f"Target encoding {te} not in the simulation."
                    assert te != encoding, "Agent cannot target its own encodings."
            else:
                raise TypeError("Target encodings must be a set or an integer.")
        self._target_mapping = value

    def get_done(self, agent, **kwarg):
        if agent.encoding not in self.target_mapping:
            return False
        active_encodings = {agent.encoding for agent in self.agents.values() if agent.active}
        target_encodings = self.target_mapping[agent.encoding]
        return False if set.intersection(active_encodings, target_encodings) else True

    def get_all_done(self, **kwargs):
        active_encodings = {agent.encoding for agent in self.agents.values() if agent.active}
        done_encodings = {
            encoding: False if set.intersection(active_encodings, target_encodings) else True
            for encoding, target_encodings in self.target_mapping.items()
        }
        if self.sim_ends_if_one_done:
            return any(done_encodings.values())
        else:
            return all(done_encodings.values())


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
