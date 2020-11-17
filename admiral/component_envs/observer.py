
from abc import ABC, abstractmethod

import numpy as np

from admiral.component_envs.world import WorldEnv, WorldAgent
from admiral.component_envs.team import TeamAgent
from admiral.envs import Agent

class ObservingAgent(Agent):
    def __init__(self, view=None, **kwargs):
        assert view is not None, "view must be nonnegative integer"
        self.view = view
        super().__init__(**kwargs)

        from gym.spaces import Box
        #TODO: num_teams = ...
        # self.observation_space['agents'] = Box(-1, num_teams, (view*2+1, view*2+1), np.int)
        # Alternatively, we can leave it unbounded...
        self.observation_space['agents'] = Box(-1, np.inf, (view*2+1, view*2+1), np.int)
        # TODO: What about channel observing agent? The parameters would be the
        # same: just view. But the observation space would be different. Do I make
        # a different type of agent or do I allow the channel environment set
        # the observation space of the agents?
        # I think the environment should take part in constructing this because
        # we have to know if the environment is Grid-based.
    
    @property
    def configured(self):
        return super().configured and self.view is not None

class ObservingTeamAgent(ObservingAgent, TeamAgent, WorldAgent):
    pass

class ObserverEnv(ABC):
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        # assert that the agents are observing agents
    
    @abstractmethod
    def get_obs(self, my_id, **kwargs):
        pass

class GridObserverEnv(ObserverEnv):
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
            if other_id == my_id: continue
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

class GridChannelObserverEnv(ObserverEnv):
    def __init__(self, region=None, agents=None, **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        # assert that the agents are TeamAgents
    
    def get_obs(self, my_id, **kwargs):
        pass
        # TODO: Do something similar to GridObserver, but add channesl for the
        # teams.