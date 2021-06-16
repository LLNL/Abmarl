
from abc import ABC

import numpy as np

from .agent import GridWorldAgent, GridObservingAgent,  MovingAgent


class GridWorldBaseComponent(ABC):
    """
    Component base class from which all components will inherit.

    The base class stores the agents dictionary. The base class also stores the
    dimensions of the grid world. It creates a 2d np array for indexing the agents
    on their positions.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
    
    @property
    def agents(self):
        """
        A dict that maps the Agent's id to the Agent object. All agents must be
        GridWorldAgents.
        """
        return self._agents

    @agents.setter
    def agents(self, value_agents):
        assert type(value_agents) is dict, "Agents must be a dict."
        for agent_id, agent in value_agents.items():
            assert isinstance(agent, GridWorldAgent), \
                "Values of agents dict must be instance of GridWorldAgent."
            assert agent_id == agent.id, \
                "Keys of agents dict must be the same as the Agent's id."
        self._agents = value_agents
