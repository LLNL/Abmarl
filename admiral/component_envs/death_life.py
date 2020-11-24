
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
        return super().configured and self.min_health is not None and self.max_health is not None

class DyingAgent(HealthAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_alive = True
    
    @property
    def configured(self):
        return super().configured and self.is_alive is not None

class DyingComponent:
    """
    Agents can die if their health falls below their minimal health value. Health
    can decrease in a number of interactions. This environemnt provides an entropy
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
                if agent.initial_health is not None:
                    agent.health = agent.initial_health
                else:
                    agent.health = np.random.uniform(agent.min_health, agent.max_health)
            if isinstance(agent, DyingAgent):
                agent.is_alive = True
    
    def apply_entropy(self, agent, **kwargs):
        """
        All agents' health decreases by the entropy.
        """
        if isinstance(agent, HealthAgent):
            agent.health -= self.entropy
    
    def process_death(self, agent, **kwargs):
        """
        Process agent's death. If the health falls below the
        the minimal value, then the agent dies.
        """
        if isinstance(agent, DyingAgent):
            if agent.health <= agent.min_health:
                agent.health = agent.min_health
                agent.is_alive = False

        # TODO: This should not be a part of proces death because it doesn't modify
        # whether the agent is alive or not. Should be a function like update
        # agent's health and handles the upper bound and can be called whenever
        # we want to change the agent's health.
        if isinstance(agent, HealthAgent):
            if agent.health > agent.max_health:
                agent.health = agent.max_health
