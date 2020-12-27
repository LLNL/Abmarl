
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.team import TeamAgent

class PositionAgent(Agent):
    """
    Agents have a position in the environment.

    starting_position (np.array):
        The desired starting position for this agent.
    """
    def __init__(self, starting_position=None, **kwargs):
        super().__init__(**kwargs)
        self.starting_position = starting_position
        self.position = None

class PositionObservingAgent(Agent):
    """
    Agents can observe other agents.

    position_view_range (int):
        Any agent within this many spaces will be fully observed.
    """
    def __init__(self, position_view_range=None, **kwargs):
        super().__init__(**kwargs)
        assert position_view_range is not None, "position_view_range must be nonnegative integer"
        self.position_view_range = position_view_range
    
    @property
    def configured(self):
        """
        Agents are configured if the position_view_range parameter is set.
        """
        return super().configured and self.position_view_range is not None

class PositionState:
    """
    Manages the agents' positions. All position updates must be within the region.

    region (int):
        The size of the environment.
    
    agents (dict):
        The dictionary of agents.
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
        """
        Set the agent's position to the incoming value only if the new position
        is within the region.
        """
        if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
            agent.position = _position
    
    def modify_position(self, agent, value, **kwargs):
        """
        Add some value to the position of the agent.
        """
        self.set_position(agent, agent.position + value)

class PositionObserver:
    """
    Observe the positions of all the agents in the simulator.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            agent.observation_space['position'] = Dict({
                other.id: Box(0, self.position.region, (2,), np.int) for other in agents.values()
            })

    def get_obs(self, *args, **kwargs):
        """
        Get the positions of all the agents in the simulator.
        """
        return {agent.id: self.agents[agent.id].position for agent in self.agents.values()}

class RelativePositionObserver:
    """
    Observe the relative positions of agents in the simulator.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents=agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Dict({
                    other.id: Box(-position.region, position.region, (2,), np.int) for other in agents.values() if other.id != agent.id
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the relative positions of all the agents in the simulator.
        """
        obs = {}
        for other in self.agents.values():
            if other.id == agent.id: continue # Don't observe your own position
            r_diff = other.position[0] - agent.position[0]
            c_diff = other.position[1] - agent.position[1]
            obs[other.id] = np.array([r_diff, c_diff])
        return obs

class GridPositionBasedObserver:
    """
    Agents observe a grid of size position_view_range centered on their
    position. The values of the cells are as such:
        Out of bounds  : -1
        Empty          :  0
        Agent occupied : 1
    
    position (PositionState):
        The position state handler, which contains the region.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Box(-1, 1, (agent.position_view_range*2+1, agent.position_view_range*2+1), np.int)

    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position.
        """
        if isinstance(my_agent, PositionObservingAgent):
            signal = np.zeros((my_agent.position_view_range*2+1, my_agent.position_view_range*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.position_view_range - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.position_view_range - my_agent.position[0], :] = -1
            if my_agent.position_view_range - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.position_view_range - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.position_view_range - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.position_view_range - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.position_view_range - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.position_view_range - 1:] = -1

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.position_view_range <= r_diff <= my_agent.position_view_range and -my_agent.position_view_range <= c_diff <= my_agent.position_view_range:
                    r_diff += my_agent.position_view_range
                    c_diff += my_agent.position_view_range
                    signal[r_diff, c_diff] = 1 # There is an agent at this location.

            return signal

class GridPositionTeamBasedObserver:
    """
    Agents observe a grid of size position_view_range centered on their
    position. The observation contains one channel per team, where the value of
    the cell is the number of agents on that team that occupy that square. -1
    indicates out of bounds.
    
    position (PositionState):
        The position state handler, which contains the region.

    team_state (TeamState):
        The team state handler, which contains the number of teams.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, team_state=None, agents=None, **kwargs):
        self.position = position
        self.team_state = team_state
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, TeamAgent)
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Box(-1, np.inf, (agent.position_view_range*2+1, agent.position_view_range*2+1, self.team_state.number_of_teams), np.int)
    
    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position. Each team has its own channel and the value represents the number
        of agents of that team occupying the same square.
        """
        if isinstance(my_agent, PositionObservingAgent) and isinstance(my_agent, TeamAgent):
            signal = np.zeros((my_agent.position_view_range*2+1, my_agent.position_view_range*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.position_view_range - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.position_view_range - my_agent.position[0], :] = -1
            if my_agent.position_view_range - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.position_view_range - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.position_view_range - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.position_view_range - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.position_view_range - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.position_view_range - 1:] = -1

            # Repeat the boundaries signal for all teams
            signal = np.repeat(signal[:, :, np.newaxis], self.team_state.number_of_teams, axis=2)

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.position_view_range <= r_diff <= my_agent.position_view_range and -my_agent.position_view_range <= c_diff <= my_agent.position_view_range:
                    r_diff += my_agent.position_view_range
                    c_diff += my_agent.position_view_range
                    signal[r_diff, c_diff, other_agent.team] += 1

            return signal
