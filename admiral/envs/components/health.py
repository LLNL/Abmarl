
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
    
    def reset(self, agent, **kwargs):
        """
        Reset the health and life state of all applicable agents.
        """
        if agent.initial_health is not None:
            agent.health = agent.initial_health
        else:
            agent.health = np.random.uniform(agent.min_health, agent.max_health)
        agent.is_alive = True
    
    def set_health(self, agent, _health):
        if _health <= agent.min_health:
            agent.health = agent.min_health
            agent.is_alive = False
        elif _health >= agent.max_health:
            agent.health = agent.max_health
        else:
            agent.health = _health
    
    def modify_health(self, agent, value):
        self.set_health(agent, agent.health + value)

    def apply_entropy(self, agent, **kwargs):
        """
        Apply entropy to the agent, decreasing its health by a small amount.
        """
        self.modify_health(agent, -self.entropy, **kwargs)

class HealthObserver:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        from gym.spaces import Dict, Box
        for agent in agents.values():
            agent.observation_space['health'] = Dict({
                other.id: Box(other.min_health, other.max_health, (1,), np.float) for other in self.agents.values()
            })
    
    def get_obs(self, *args, **kwargs):
        return {agent.id: self.agents[agent.id].health for agent in self.agents.values()}

class LifeObserver:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        from gym.spaces import Dict, Box
        for agent in agents.values():
            agent.observation_space['life'] = Dict({
                other.id: Box(0, 1, (1,), np.int) for other in self.agents.values()
            })
    
    def get_obs(self, *args, **kwargs):
        return {agent.id: self.agents[agent.id].is_alive for agent in self.agents.values()}
