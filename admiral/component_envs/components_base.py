
from abc import ABC, abstractmethod

import numpy as np

class World(ABC):
    """
    World is an abstract notion for some space in which agents exist. It is
    defined the set of agents that exist in it and the bounds of the world.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        self.agents = kwargs['agents'] if 'agents' in kwargs else {}

    @abstractmethod
    def reset(self, **kwargs):
        """
        Place agents throughout the world.
        """
        pass

    @abstractmethod
    def render(self, **kwargs):
        """
        Draw the world with the agents.
        """
        pass

class Movement(ABC):
    def __init__(self, region=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        self.agents = kwargs['agents'] if 'agents' in kwargs else {}
        # TODO: Do we require agents in this component?

    @abstractmethod
    def process_move(self, agent, move, **kwargs):
        """
        Move the agent some amount.
        """
        pass

class Resources(ABC):
    """
    Resources exist in space in the world and can typically grow, be harvested, and spread. Spread
    is treated by child classes under the regrow function.
    """
    def __init__(self, region=None, coverage=0.75, min_value=0.1, max_value=1.0, regrow_rate=0.04, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.agents = kwargs['agents'] if 'agents' in kwargs else {}
        self.region = region
        self.min_value = min_value
        self.max_value = max_value
        self.regrow_rate = regrow_rate
        self.coverage = coverage
    
    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the initial state of the resources.
        """
        pass

    @abstractmethod
    def process_harvest(self, location, amount, **kwargs):
        """
        Harvest a certain amount of resources at this location.

        Return the amount that was actually harvested.
        """
        pass

    @abstractmethod
    def regrow(self, **kwargs):
        """
        Process the regrowth, which is done according to the revival rate.
        """
        pass

class Health(ABC):
    """
    Agents can have health, and this component helps process changes in their health based on their
    actions and the actions of other agents. If the health falls below the agent's threshold, then
    the agent dies.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "Agents must be a dict."
        self.agents = agents

    def reset(self, **kwargs):
        """
        Reset the agent's health values.
        """
        for agent in self.agents.values():
            # TODO: use max and min health specified in the environment.
            agent.health = np.random.uniform(0.5, 1.0)
            agent.is_alive = True

    def process_death(self, agent, **kwargs):
        """
        If the agent's health falls below its threshold, then it dies.
        """
        if agent.health < agent.death:
            agent.is_alive = False

class Reproducer(ABC):
    """
    Some agents can reproduce; this component helps process that reproduction.
    """
    def __init__(self, agents=None, **kwargs):
        assert type(agents) is dict, "Agents must be a dict."
        self.agents = agents
        for agent in self.agents.values():
            agent.is_original = True
        self.agent_counter = len(self.agents)

    def reset(self, **kwargs):
        """
        Reset the set of agents in the world to the original set.
        """
        for agent_id, agent in list(self.agents.items()):
            if not agent.is_original:
                del self.agents[agent_id]
        self.agent_counter = len(self.agents)

    @abstractmethod
    def process_reproduce(self, **kwargs):
        """
        Process the agents reproducing.
        """
        pass
