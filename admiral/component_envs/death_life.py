
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent

class HealthAgent(Agent):
    def __init__(self, min_health=0.0, max_health=1.0, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_health = initial_health
        self.min_health = min_health
        self.max_health = max_health
        self.health = None
    
    @property
    def configured(self):
        return super().configured and self.health is not None

class DyingAgent(HealthAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_alive = True
    
    @property
    def configured(self):
        return super().configured and self.is_alive is not None

class DyingEnv:
    """
    Agents can die if their health falls below their minimal health value. Health
    can decrease in a number of interactions. This environemnt provides an entropy
    that indicates how much health the agent loses in each step.

    Provides the process_death and apply_entropy api.
    """
    def __init__(self, agents=None, entropy=0.1, **kwargs):
        assert type(agents) is dict, "Agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, DyingAgent)
        self.agents = agents
        self.entropy = entropy
    
    def reset(self, **kwargs):
        for agent in self.agents.values():
            if agent.initial_health is not None:
                agent.health = agent.initial_health
            else:
                agent.health = np.random.uniform(agent.min_health, agent.max_health)
            agent.is_alive = True
    
    def apply_entropy(self, agent, **kwargs):
        """
        All agents' health decreases by the entropy.
        """
        agent.health -= self.entropy
    
    def process_death(self, agent, **kwargs):
        """
        Process agent's death. If the health falls below the
        the minimal value, then the agent dies.
        """
        if agent.health <= agent.min_health:
            agent.health = agent.min_health
            agent.is_alive = False
        if agent.health > agent.max_health:
            agent.health = agent.max_health

    def render(self, **kwargs):
        for agent in self.agents.values():
            print(f'{agent.id}: {agent.health}, {agent.is_alive}')
