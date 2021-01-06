
import numpy as np

from admiral.envs.components.agent import LifeAgent
from admiral.envs.components.state import LifeState


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
