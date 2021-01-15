
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs.components.agent import LifeAgent, PositionAgent, SpeedAngleAgent

# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class LifeState:
    """
    Agents can die if their health falls below their minimal health value. Health
    can decrease in a number of interactions. This environment provides an entropy
    that indicates how much health an agent loses when apply_entropy is called.
    This is a generic entropy for the step. If you want to specify health changes
    for specific actions, such as being attacked or harvesting, you must write
    it in the environment.

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
                agent.health = 0
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

class PositionState(ABC):
    """
    Manages the agents' positions.

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
                if agent.initial_position is not None:
                    agent.position = agent.initial_position
                else:
                    self.random_reset(agent)
    
    @abstractmethod
    def random_reset(self, agent, **kwargs):
        """
        Reset the agents' positions. Child classes implement this according to their
        specs. For example, GridPositionState assigns random integers as the position,
        whereas ContinuousPositionState assigns random numbers.
        """
        pass

    @abstractmethod
    def set_position(self, agent, position, **kwargs):
        """
        Set the position of the agents. Child classes implement.
        """
        pass
    
    def modify_position(self, agent, value, **kwargs):
        """
        Add some value to the position of the agent.
        """
        if isinstance(agent, PositionAgent):
            self.set_position(agent, agent.position + value)

class GridPositionState(PositionState):
    """
    Agents are positioned in a grid and cannot go outside of the region.
    """
    def set_position(self, agent, _position, **kwargs):
        """
        Set the agent's position to the incoming value only if the new position
        is within the region.
        """
        if isinstance(agent, PositionAgent):
            if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
                agent.position = _position

    def random_reset(self, agent, **kwargs):
        """
        Set the agents' random positions as integers within the region.
        """
        agent.position = np.random.randint(0, self.region, 2)

class ContinuousPositionState(PositionState):
    """
    Agents are positioned in a continuous space and can go outside the bounds
    of the region.
    """
    def set_position(self, agent, _position, **kwargs):
        """
        Set the agent's position to the incoming value.
        """
        if isinstance(agent, PositionAgent):
            agent.position = _position

    def random_reset(self, agent, **kwargs):
        """
        Set the agents' random positions as numbers within the region.
        """
        agent.position = np.random.uniform(0, self.region, 2)

class SpeedAngleState:
    """
    Manges the agents' speed, banking angles, and ground angles.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
    
    def reset(self, **kwargs):
        """
        Reset the agents' speeds and ground angles.
        """
        for agent in self.agents.values():
            if isinstance(agent, SpeedAngleAgent):
                # Reset agent speed
                if agent.initial_speed is not None:
                    agent.speed = agent.initial_speed
                else:
                    agent.speed = np.random.uniform(agent.min_speed, agent.max_speed)

                # Reset agent banking angle
                if agent.initial_banking_angle is not None:
                    agent.banking_angle = agent.initial_banking_angle
                else:
                    agent.banking_angle = np.random.uniform(-agent.max_banking_angle, agent.max_banking_angle)

                # Reset agent ground angle
                if agent.initial_ground_angle is not None:
                    agent.ground_angle = agent.initial_ground_angle
                else:
                    agent.ground_angle = np.random.uniform(0, 360)
    
    def set_speed(self, agent, _speed, **kwargs):
        """
        Set the agent's speed if it is between its min and max speed.
        """
        if isinstance(agent, SpeedAngleAgent):
            if agent.min_speed <= _speed <= agent.max_speed:
                agent.speed = _speed
    
    def modify_speed(self, agent, value, **kwargs):
        """
        Modify the agent's speed.
        """
        if isinstance(agent, SpeedAngleAgent):
            self.set_speed(agent, agent.speed + value)
    
    def set_banking_angle(self, agent, _banking_angle, **kwargs):
        """
        Set the agent's banking angle if it is between its min and max angle.
        """
        if isinstance(agent, SpeedAngleAgent):
            if abs(_banking_angle) <= agent.max_banking_angle:
                agent.banking_angle = _banking_angle
                self.modify_ground_angle(agent, agent.banking_angle)
    
    def modify_banking_angle(self, agent, value, **kwargs):
        """
        Modify the agent's banking angle.
        """
        if isinstance(agent, SpeedAngleAgent):
            self.set_banking_angle(agent, agent.banking_angle + value)

    def set_ground_angle(self, agent, _ground_angle, **kwargs):
        """
        Set the agent's ground angle, which will be modded to fall between 0 and
        360.
        """
        if isinstance(agent, SpeedAngleAgent):
            agent.ground_angle = _ground_angle % 360
    
    def modify_ground_angle(self, agent, value, **kwargs):
        """
        Modify the agent's ground angle.
        """
        if isinstance(agent, SpeedAngleAgent):
            self.set_ground_angle(agent, agent.ground_angle + value)


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

    An agent can harvest up to its max harvest value on the cell it occupies. It
    can observe the resources in a grid surrounding its position, up to its view
    distance.

    agents (dict):
        The dictionary of agents.

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
    
    initial_resources (np.array):
        Instead of specifying the above resource-related parameters, we can provide
        an initial state of the resources. At reset time, the resources will be
        set to these original resources. Otherwise, the resources will be set
        to random values between the min and max value up to some coverage of the
        region.
    """
    def __init__(self, agents=None, region=None, coverage=0.75, min_value=0.1, max_value=1.0,
            regrow_rate=0.04, initial_resources=None, **kwargs):        
        self.initial_resources = initial_resources
        if self.initial_resources is None:
            assert type(region) is int, "Region must be an integer."
            self.region = region
        else:
            self.region = self.initial_resources.shape[0]
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
        if self.initial_resources is not None:
            self.resources = self.initial_resources
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
