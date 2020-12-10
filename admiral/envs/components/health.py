
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
    
    @property
    def configured(self):
        """
        The agent is successfully configured if is_alive is not None.
        """
        return super().configured and self.is_alive is not None

class LifeState:
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
            if agent.initial_health is not None:
                agent.health = agent.initial_health
            else:
                agent.health = np.random.uniform(agent.min_health, agent.max_health)
            agent.is_alive = True
    
    def set_health(self, agent_id, _health):
        if _health <= self.min_health:
            self.health = self.min_health
            self.is_alive = False
        elif _health >= self.max_health:
            self.health = self.max_health
        else:
            self.health = _health
    
    def modify_health(self, agent_id, value):
        self.set_health(agent_id, self.agents[agent_id].health + value)

    def apply_entropy(self, agent_id, **kwargs):
        """
        Apply entropy to the agent, decreasing its health by a small amount.
        """
        self.agents['agent_id'].add_health(-self.entropy)

class HealthObserver:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        from gym.spaces import Dict, Box
        for agent in agent.values():
            agent.observation_space['health'] = Dict({
                other.id: Box(other.min_health, other.max_health, (1,), np.float) for other in self.agents
            })
    
    def get_state(self, *args, **kwargs):
        return {agent.id: self.agents[agent.id].health for agent in self.agents}

class LifeObserver:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        from gym.spaces import Dict, Box
        for agent in agent.values():
            agent.observation_space['life'] = Dict({
                other.id: Box(0, 1, (1,), np.int) for other in self.agents
            })
    
    def get_state(self, *args, **kwargs):
        return {agent.id: self.agents[agent.id].is_alive for agent in self.agents}
