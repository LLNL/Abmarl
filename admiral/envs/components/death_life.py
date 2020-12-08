
import numpy as np

from admiral.envs import Agent

class HealthAgent(Agent):
    """
    Agents have health.

    min_health (float):
        The minimum value the health can reach.
    
    max_health (float):
        The maximum value the health can reach.

    initial_health (float):
        The initial health of the agent. The health will be set to this initial
        option at reset time.

    Note: You should use the set health function below to intelligently set the
    agent's health.
    """
    def __init__(self, min_health=0.0, max_health=1.0, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        if initial_health is not None:
            assert min_health <= initial_health <= max_health
        self.initial_health = initial_health
        self.min_health = min_health
        self.max_health = max_health
        self.health = None

    def get_health(self):
        """
        Standard getter for the health parameter.
        """
        return self.health
    
    def set_health(self, health_in):
        """
        Set the agent's health to the incoming value. Ensures that the health is
        between the minimum and maximum values at all times.
        """
        if health_in <= self.min_health:
            self.health = self.min_health
        elif health_in >= self.max_health:
            self.health = self.max_health
        else:
            self.health = health_in
    
    def reset_health(self):
        """
        Reset the agent's health. If initial_health was specified when the agent
        was created, then the health will be reset to that value. Otherwise, the
        health is reset to a value between the min and max.
        """
        if self.initial_health is not None:
            self.health = self.initial_health
        else:
            self.health = np.random.uniform(self.min_health, self.max_health)
    
    def add_health(self, add_health_amount):
        """
        Convenience function for incrementing the health by the incoming value.
        """
        self.set_health(self.health + add_health_amount)

    @property
    def configured(self):
        """
        The agent is successfully configured if the min and max health are specified.
        """
        return super().configured and self.min_health is not None and self.max_health is not None

class LifeAgent(HealthAgent):
    """
    Agents have health and a parameter that indicates if the agent is alive or
    dead.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_alive = True
    
    def set_health(self, health_in):
        """
        In addition to keeping the health between min and max value, we will update
        the alive status of the agent here. If the health falls below the min_value,
        the agent dies.
        """
        super().set_health(health_in)
        if health_in <= self.min_health:
            self.is_alive = False
        
    def reset_life(self):
        """
        Reset the agent to alive
        """
        self.is_alive = True
    
    @property
    def configured(self):
        """
        The agent is successfully configured if is_alive is not None.
        """
        return super().configured and self.is_alive is not None

class DyingComponent:
    """
    Agents can die if their health falls below their minimal health value. Health
    can decrease in a number of interactions. This environment provides an entropy
    that indicates how much health the agent loses in each step.

    agents (dict):
        Dictionary of agents.
    
    entropy (float):
        The amount of health that is depleted from the agents whenever apply_entropy
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
            if isinstance(agent, HealthAgent):
                agent.reset_health()
            if isinstance(agent, LifeAgent):
                agent.reset_life()
    
    def apply_entropy(self, agent, **kwargs):
        """
        Apply entropy to the agent, decreasing its health by a small amount.
        """
        if isinstance(agent, HealthAgent):
            agent.add_health(-self.entropy)
