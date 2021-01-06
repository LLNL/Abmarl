
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.agent import TeamAgent

from admiral.envs.components.state import TeamState


class TeamObserver:
    """
    Observe the team of each agent in the simulator.
    """
    def __init__(self, team=None, agents=None, **kwargs):
        self.team = team
        self.agents = agents
    
        from gym.spaces import Box, Dict
        for agent in agents.values():
            agent.observation_space['team'] = Dict({
                other.id: Box(-1, self.team.number_of_teams, (1,), np.int) for other in agents.values() if isinstance(other, TeamAgent)
            })
    
    def get_obs(self, *args, **kwargs):
        """
        Get the team of each agent in the simulator.
        """
        return {'team': {other.id: self.agents[other.id].team for other in self.agents.values() if isinstance(other, TeamAgent)}}
    
    @property
    def null_value(self):
        return -1
