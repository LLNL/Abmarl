
import numpy as np

from admiral.envs.components.agent import LifeAgent, PositionAgent

# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class LifeState:
    """
    Agents can die if their health falls below their minimal health value. Health
    can decrease in a number of interactions. This environment provides an entropy
    that indicates how much health an agent loses when apply_entropy is called.

    agents (dict):
        Dictionary of agents.
    
    entropy (float):
        The amount of health that is depleted from an agent whenever apply_entropy
        is called.
    """
    def __init__(self, agents=None, entropy=0.1, **kwargs):
        assert type(agents) is dict, "Agents must be a dict"
        self.agents = agents
        self.entropy = entropy
    
    def reset(self, **kwargs):
        """
        Reset the health and life state of all applicable agents.
        """
        for agent in self.agents.values():
            if isinstance(agent, LifeAgent):
                if agent.initial_health is not None:
                    agent.health = agent.initial_health
                else:
                    agent.health = np.random.uniform(agent.min_health, agent.max_health)
                agent.is_alive = True
    
    def set_health(self, agent, _health):
        """
        Set the health of an agent to a specific value, bounded by the agent's
        min and max health-value. If that value is less than the agent's health,
        then the agent dies.
        """
        if isinstance(agent, LifeAgent):
            if _health <= agent.min_health:
                agent.health = agent.min_health
                agent.is_alive = False
            elif _health >= agent.max_health:
                agent.health = agent.max_health
            else:
                agent.health = _health
    
    def modify_health(self, agent, value):
        """
        Add some value to the health of the agent.
        """
        if isinstance(agent, LifeAgent):
            self.set_health(agent, agent.health + value)

    def apply_entropy(self, agent, **kwargs):
        """
        Apply entropy to the agent, decreasing its health by a small amount.
        """
        if isinstance(agent, LifeAgent):
            self.modify_health(agent, -self.entropy, **kwargs)



# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class PositionState:
    """
    Manages the agents' positions. All position updates must be within the region.

    region (int):
        The size of the environment.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

    def reset(self, **kwargs):
        """
        Reset the agents' positions. If the agents were created with a starting
        position, then use that. Otherwise, randomly assign a position in the region.
        """
        for agent in self.agents.values():
            if isinstance(agent, PositionAgent):
                if agent.starting_position is not None:
                    agent.position = agent.starting_position
                else:
                    agent.position = np.random.randint(0, self.region, 2)
    
    def set_position(self, agent, _position, **kwargs):
        """
        Set the agent's position to the incoming value only if the new position
        is within the region.
        """
        if isinstance(agent, PositionAgent):
            if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
                agent.position = _position
    
    def modify_position(self, agent, value, **kwargs):
        """
        Add some value to the position of the agent.
        """
        if isinstance(agent, PositionAgent):
            self.set_position(agent, agent.position + value)



# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class GridResourceState:
    """
    Resources exist in the cells of the grid. The grid is populated with resources
    between the min and max value on some coverage of the region at reset time.
    If original resources is specified, then reset will set the resources back 
    to that original value. This component supports resource depletion: if a resource falls below
    the minimum value, it will not regrow. Agents can harvest resources from the cell they occupy.
    Agents can observe the resources in a grid-like observation surrounding their positions.

    The action space of GridResourcesHarvestingAgents is appended with
    Box(0, agent.max_harvest, (1,), np.float), indicating that the agent can harvest
    up to its max harvest value on the cell it occupies.

    The observation space of ObservingAgents is appended with
    Box(0, self.max_value, (agent.view*2+1, agent.view*2+1), np.float), indicating
    that an agent can observe the resources in a grid surrounding its position,
    up to its view distance.

    agents (dict):
        The dictionary of agents. Because agents harvest and observe resources
        based on their positions, agents must be GridPositionAgents.

    region (int):
        The size of the region

    coverage (float):
        The ratio of the region that should start with resources.

    min_value (float):
        The minimum value a resource can have before it cannot grow back. This is
        different from the absolute minimum value, 0, which indicates that there
        are no resources in the cell.
    
    max_value (float):
        The maximum value a resource can have.

    regrow_rate (float):
        The rate at which resources regrow.
    
    original_resources (np.array):
        Instead of specifying the above resource-related parameters, we can provide
        an initial state of the resources. At reset time, the resources will be
        set to these original resources. Otherwise, the resources will be set
        to random values between the min and max value up to some coverage of the
        region.
    """
    def __init__(self, agents=None, region=None, coverage=0.75, min_value=0.1, max_value=1.0,
            regrow_rate=0.04, original_resources=None, **kwargs):        
        self.original_resources = original_resources
        if self.original_resources is None:
            assert type(region) is int, "Region must be an integer."
            self.region = region
        else:
            self.region = self.original_resources.shape[0]
        self.min_value = min_value
        self.max_value = max_value
        self.regrow_rate = regrow_rate
        self.coverage = coverage

        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

    def reset(self, **kwargs):
        """
        Reset the resources. If original resources is specified, then the resources
        will be reset back to this original value. Otherwise, the resources will
        be randomly generated values between the min and max value up to some coverage
        of the region.
        """
        if self.original_resources is not None:
            self.resources = self.original_resources
        else:
            coverage_filter = np.zeros((self.region, self.region))
            coverage_filter[np.random.uniform(0, 1, (self.region, self.region)) < self.coverage] = 1.
            self.resources = np.multiply(
                np.random.uniform(self.min_value, self.max_value, (self.region, self.region)),
                coverage_filter
            )
    
    def set_resources(self, location, value, **kwargs):
        """
        Set the resource at a certain location to a value, bounded between 0 and
        the maximum resource value.
        """
        assert type(location) is tuple
        if value <= 0:
            self.resources[location] = 0
        elif value >= self.max_value:
            self.resources[location] = self.max_value
        else:
            self.resources[location] = value
    
    def modify_resources(self, location, value, **kwargs):
        """
        Add some value to the resource at a certain location.
        """
        assert type(location) is tuple
        self.set_resources(location, self.resources[location] + value, **kwargs)

    def regrow(self, **kwargs):
        """
        Regrow the resources according to the regrow_rate.
        """
        self.resources[self.resources >= self.min_value] += self.regrow_rate
        self.resources[self.resources >= self.max_value] = self.max_value



# ------------ #
# --- Team --- #
# ------------ #

class TeamState:
    """
    Team state manages the state of agents' teams. Since these are not changing,
    there is not much to manage. It really just keeps track of the number_of_teams.

    number_of_teams (int):
        The number of teams in this simulation.
    """
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        self.number_of_teams = number_of_teams
        self.agents = agents
