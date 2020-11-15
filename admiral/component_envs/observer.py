
from abc import ABC, abstractmethod

import numpy as np

from admiral.component_envs.world import WorldEnv, WorldAgent
from admiral.envs import Agent

class TeamAgent(Agent):
    """
    Team should be an integer.
    """
    def __init__(self, team=None, **kwargs):
        super().__init__(**kwargs)
        self.team = team
    
    @property
    def configured(self):
        return super().configured and self.team is not None

class ObservingAgent(Agent):
    def __init__(self, view=None, **kwargs):
        super().__init__(**kwargs)
        self.view = view
    
    @property
    def configured(self):
        return super().configured and self.view is not None

class ObservingTeamAgent(ObservingAgent, TeamAgent, WorldAgent):
    pass

class Observer(ABC):
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        # assert that the agents are observing agents
    
    @abstractmethod
    def get_obs(self, my_id, **kwargs):
        pass

class GridObserver(Observer):
    def __init__(self, region=None, agents=None, **kwargs):
        self.region = region
        self.agents = agents
        # Agents must have teams, views, and positions.
    
    def get_obs(self, my_id, **kwargs):
        """
        These cells are filled with the value of the agent's type, including -1
        for out of bounds and 0 for empty square. If there are multiple agents
        on the same cell, then we prioritize the agent that is of a different
        type. For example, a prey will only see a predator on a cell that a predator
        and another prey both occupy.
        """
        my_agent = self.agents[my_id]
        signal = np.zeros((my_agent.view*2+1, my_agent.view*2+1))

        # --- Determine the boundaries of the agents' grids --- #
        # For left and top, we just do: view - x,y >= 0
        # For the right and bottom, we just do region - x,y - 1 - view > 0
        if my_agent.view - my_agent.position[0] >= 0: # Top end
            signal[0:my_agent.view - my_agent.position[0], :] = -1
        if my_agent.view - my_agent.position[1] >= 0: # Left end
            signal[:, 0:my_agent.view - my_agent.position[1]] = -1
        if self.region - my_agent.position[0] - my_agent.view - 1 < 0: # Bottom end
            signal[self.region - my_agent.position[0] - my_agent.view - 1:,:] = -1
        if self.region - my_agent.position[1] - my_agent.view - 1 < 0: # Right end
            signal[:, self.region - my_agent.position[1] - my_agent.view - 1:] = -1

        # --- Determine the positions of all the other alive agents --- #
        for other_id, other_agent in self.agents.items():
            # if other_id == my_id or other_id in self.cemetery: continue
            r_diff = other_agent.position[0] - my_agent.position[0]
            c_diff = other_agent.position[1] - my_agent.position[1]
            if -my_agent.view <= r_diff <= my_agent.view and -my_agent.view <= c_diff <= my_agent.view:
                r_diff += my_agent.view
                c_diff += my_agent.view
                if signal[r_diff, c_diff] != 0: # Already another agent here
                    if type(my_agent) != type(other_agent):
                        signal[r_diff, c_diff] = other_agent.team
                else:
                    signal[r_diff, c_diff] = other_agent.team
        
        return signal

class GridChannelObserver(Observer):
    def __init__(self, region=None, agents=None, **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        # assert that the agents are TeamAgents
    
    def get_obs(self, my_id, **kwargs):
        pass
        # TODO: Do something similar to GridObserver, but add channesl for the
        # teams.