
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.team import TeamAgent
from admiral.envs.components.observer import ObservingAgent

class PositionAgent(Agent):
    def __init__(self, starting_position=None, **kwargs):
        super().__init__(**kwargs)
        self.starting_position = starting_position
        self.position = None

class PositionState:
    """
    Manages the agents' positions.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
        self.agents = agents

    def reset(self, **kwargs):
        """
        Reset the agents' positions. If the agents were created with a starting
        position, then use that. Otherwise, randomly assign a position in the region.
        """
        for agent in self.agents.values():
            if agent.starting_position is not None:
                agent.position = agent.starting_position
            else:
                agent.position = np.random.randint(0, self.region, 2)
    
    def set_position(self, agent, _position, **kwargs):
        if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
            agent.position = _position
    
    def modify_position(self, agent, value, **kwargs):
        self.set_position(agent, agent.position + value)

class PositionObserver:
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            agent.observation_space['position'] = Dict({
                other.id: Box(0, self.position.region, (2,), np.int) for other in agents.values()
            })

    def get_obs(self, *args, **kwargs):
        return {agent.id: self.agents[agent.id].position for agent in self.agents.values()}

class GridPositionBasedObserver:
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, ObservingAgent):
                agent.observation_space['position'] = Box(-1, 1, (agent.view*2+1, agent.view*2+1), np.int)

    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position.
        """
        if isinstance(my_agent, ObservingAgent):
            signal = np.zeros((my_agent.view*2+1, my_agent.view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.view - my_agent.position[0], :] = -1
            if my_agent.view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.view - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.view - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.view - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.view - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.view - 1:] = -1

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.view <= r_diff <= my_agent.view and -my_agent.view <= c_diff <= my_agent.view:
                    r_diff += my_agent.view
                    c_diff += my_agent.view
                    signal[r_diff, c_diff] = 1 # There is an agent at this location.

            return signal

class GridPositionTeamBasedObserver:
    def __init__(self, position=None, agents=None, number_of_teams=None, **kwargs):
        self.position = position
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, TeamAgent)
        self.agents = agents
        self.number_of_teams = number_of_teams

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, ObservingAgent):
                agent.observation_space['position'] = Box(-1, np.inf, (agent.view*2+1, agent.view*2+1, number_of_teams), np.int)
    
    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position. Each team has its own channel and the value represents the number
        of agents of that team occupying the same square.
        """
        if isinstance(my_agent, ObservingAgent) and isinstance(my_agent, TeamAgent):
            signal = np.zeros((my_agent.view*2+1, my_agent.view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.view - my_agent.position[0], :] = -1
            if my_agent.view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.view - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.view - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.view - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.view - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.view - 1:] = -1

            # Repeat the boundaries signal for all teams
            signal = np.repeat(signal[:, :, np.newaxis], self.number_of_teams, axis=2)

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.view <= r_diff <= my_agent.view and -my_agent.view <= c_diff <= my_agent.view:
                    r_diff += my_agent.view
                    c_diff += my_agent.view
                    signal[r_diff, c_diff, other_agent.team] += 1

            return signal
