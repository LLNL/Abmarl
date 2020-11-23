
from abc import ABC, abstractmethod

import numpy as np

from admiral.envs import Agent

def HealthAgent(min_health=0.0, max_health=1.0, initial_health=None, **kwargs):
    return {
        **Agent(**kwargs),
        'min_health': min_health,
        'max_health': max_health,
        'initial_health': initial_health,
    }

def DyingAgent(**kwargs):
    return {
        **HealthAgent(**kwargs),
        'is_alive': True,
    }

class DyingEnv:
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
            if 'min_health' in agent and 'max_health' in agent and 'initial_health' in agent:
                if agent.initial_health is not None:
                    agent.health = agent.initial_health
                else:
                    agent.health = np.random.uniform(agent.min_health, agent.max_health)
            if 'min_health' in agent and 'max_health' in agent and 'initial_health' in agent and 'is_alive' in agent:
                agent.is_alive = True
    
    def apply_entropy(self, agent, **kwargs):
        """
        All agents' health decreases by the entropy.
        """
        if 'min_health' in agent and 'max_health' in agent and 'initial_health' in agent:
            agent.health -= self.entropy
    
    def process_death(self, agent, **kwargs):
        """
        Process agent's death. If the health falls below the
        the minimal value, then the agent dies.
        """
        if 'min_health' in agent and 'max_health' in agent and 'initial_health' in agent and 'is_alive' in agent:
            if agent.health <= agent.min_health:
                agent.health = agent.min_health
                agent.is_alive = False

        # TODO: This should not be a part of proces death because it doesn't modify
        # whether the agent is alive or not. Should be a function like update
        # agent's health and handles the upper bound and can be called whenever
        # we want to change the agent's health.
        if 'min_health' in agent and 'max_health' in agent and 'initial_health' in agent:
            if agent.health > agent.max_health:
                agent.health = agent.max_health
