
import numpy as np

from admiral.envs import Agent

class LifeAgent(Agent):
    """
    Agents have health and are alive or dead.

    min_health (float):
        The minimum value the health can reach before the agent dies.
    
    max_health (float):
        The maximum value the health can reach.

    initial_health (float):
        The initial health of the agent. The health will be set to this initial
        option at reset time.
    """
    def __init__(self, min_health=0.0, max_health=1.0, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        if initial_health is not None:
            assert min_health <= initial_health <= max_health
        self.initial_health = initial_health
        self.min_health = min_health
        self.max_health = max_health
        self.is_alive = True
        self.health = None

    @property
    def configured(self):
        """
        The agent is successfully configured if the min and max health are specified
        and if is_alive is specified.
        """
        return super().configured and self.min_health is not None and self.max_health is not None and self.is_alive is not None

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

class HealthObserver:
    """
    Observe the health state of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        from gym.spaces import Dict, Box
        for agent in agents.values():
            agent.observation_space['health'] = Dict({
                other.id: Box(-1, other.max_health, (1,), np.float) for other in self.agents.values() if isinstance(other, LifeAgent)
            })
    
    def get_obs(self, *args, **kwargs):
        """
        Get the health state of all the agents in the simulator.
        """
        return {'health': {agent.id: agent.health for agent in self.agents.values() if isinstance(agent, LifeAgent)}}
    
    @property
    def null_value(self):
        return -1

class LifeObserver:
    """
    Observe the life state of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        from gym.spaces import Dict, Box
        for agent in agents.values():
            agent.observation_space['life'] = Dict({
                other.id: Box(-1, 1, (1,), np.int) for other in self.agents.values() if isinstance(other, LifeAgent)
            })
    
    def get_obs(self, *args, **kwargs):
        """
        Get the life state of all the agents in the simulator.
        """
        return {'life': {agent.id: agent.is_alive for agent in self.agents.values() if isinstance(agent, LifeAgent)}}
    
    @property
    def null_value(self):
        return -1
