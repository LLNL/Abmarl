
import numpy as np

from admiral.envs import Agent

class HealthAgent(Agent):
    def __init__(self, min_health=0.0, max_health=1.0, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        if initial_health is not None:
            assert min_health <= initial_health <= max_health
        self.initial_health = initial_health
        self.min_health = min_health
        self.max_health = max_health
        self.health = None

    def get_health(self):
        return self.health
    
    def set_health(self, health_in):
        if health_in <= self.min_health:
            self.health = self.min_health
        elif health_in >= self.max_health:
            self.health = self.max_health
        else:
            self.health = health_in
    
    def reset_health(self):
        if self.initial_health is not None:
            self.health = self.initial_health
        else:
            self.health = np.random.uniform(self.min_health, self.max_health)
    
    def add_health(self, add_health_amount):
        self.set_health(self.health + add_health_amount)

    @property
    def configured(self):
        return super().configured and self.min_health is not None and self.max_health is not None

class LifeAgent(HealthAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_alive = True
    
    def set_health(self, health_in):
        super().set_health(health_in)
        if health_in <= self.min_health:
            self.is_alive = False
        
    def reset_life(self):
        self.is_alive = True
    
    @property
    def configured(self):
        return super().configured and self.is_alive is not None

class DyingComponent:
    """
    Agents can die if their health falls below their minimal health value. Health
    can decrease in a number of interactions. This environment provides an entropy
    that indicates how much health the agent loses in each step.

    Provides the process_death and apply_entropy api.
    """
    def __init__(self, agents=None, entropy=0.1, **kwargs):
        assert type(agents) is dict, "Agents must be a dict"
        self.agents = agents
        self.entropy = entropy
    
    def reset(self, **kwargs):
        for agent in self.agents.values():
            if isinstance(agent, HealthAgent):
                agent.reset_health()
            if isinstance(agent, LifeAgent):
                agent.reset_life()
    
    def apply_entropy(self, agent, **kwargs):
        """
        All agents' health decreases by the entropy.
        """
        if isinstance(agent, HealthAgent):
            agent.add_health(-self.entropy)
