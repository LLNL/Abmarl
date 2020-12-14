
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

    def reset(self, agent, **kwargs):
        """
        Reset the agents' positions. If the agents were created with a starting
        position, then use that. Otherwise, randomly assign a position in the region.
        """
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



#
# The below can be incorporated into observers
#

#     def get_obs(self, my_id, **kwargs):
#         """
#         Generate an observation of other agents in the grid surrounding this agent's
#         position.
#         """
#         my_agent = self.agents[my_id]
#         if isinstance(my_agent, ObservingAgent):
#             signal = np.zeros((my_agent.view*2+1, my_agent.view*2+1))

#             # --- Determine the boundaries of the agents' grids --- #
#             # For left and top, we just do: view - x,y >= 0
#             # For the right and bottom, we just do region - x,y - 1 - view > 0
#             if my_agent.view - my_agent.position[0] >= 0: # Top end
#                 signal[0:my_agent.view - my_agent.position[0], :] = -1
#             if my_agent.view - my_agent.position[1] >= 0: # Left end
#                 signal[:, 0:my_agent.view - my_agent.position[1]] = -1
#             if self.region - my_agent.position[0] - my_agent.view - 1 < 0: # Bottom end
#                 signal[self.region - my_agent.position[0] - my_agent.view - 1:,:] = -1
#             if self.region - my_agent.position[1] - my_agent.view - 1 < 0: # Right end
#                 signal[:, self.region - my_agent.position[1] - my_agent.view - 1:] = -1

#             # --- Determine the positions of all the other alive agents --- #
#             for other_id, other_agent in self.agents.items():
#                 if other_id == my_id: continue # Don't observe yourself
#                 r_diff = other_agent.position[0] - my_agent.position[0]
#                 c_diff = other_agent.position[1] - my_agent.position[1]
#                 if -my_agent.view <= r_diff <= my_agent.view and -my_agent.view <= c_diff <= my_agent.view:
#                     r_diff += my_agent.view
#                     c_diff += my_agent.view
#                     signal[r_diff, c_diff] = 1 # There is an agent at this location.

#             return signal

# class GridPositionTeamsComponent(GridPositionComponent):
#     """
#     GridPositionComponent assigns the agents positions at reset and can generate
#     observations of the other agents in a grid-based view centered around the agent.
#     If the agent was created with a starting position, then that starting position
#     is applied when the simulation resets. Otherwise, the agent is assigned some
#     random position in the region.

#     The agents observation space is appended with
#     Box(-1, inf, (agent.view*2+1, agent.view*2+1, number_of_teams), np.int),
#     indicating that the agents can see a grid of size view*2+1 surrounding its
#     current location. The view contains one channel for each team, and the values
#     in those channels are -1 if out of bounds, 0 if empty, and otherwise it is
#     the number of agents of that team occupying that cell.
    
#     region (int):
#         The size of the region. Agents' positions can be anywhere within this region.
    
#     agents (dict):
#         The dictionary of agents. Any agent that is in the GridPositionComponent
#         must be a GridPositionAgent. Because the observations are channeled by
#         the agents' teams, every agent must be a TeamAgent.
    
#     number_of_teams (int):
#         The fixed number of teams in this simulation.
#     """
#     def __init__(self, region=None, agents=None, number_of_teams=None, **kwargs):
#         self.region = region
#         for agent in agents.values():
#             assert isinstance(agent, GridPositionAgent)
#             assert isinstance(agent, TeamAgent)
#         self.agents = agents
#         self.number_of_teams = number_of_teams

#         from gym.spaces import Box
#         for agent in self.agents.values():
#             if isinstance(agent, ObservingAgent):
#                 agent.observation_space['agents'] = Box(-1, np.inf, (agent.view*2+1, agent.view*2+1, number_of_teams), np.int)
    
#     def get_obs(self, my_id, **kwargs):
#         """
#         Generate an observation of other agents in the grid surrounding this agent's
#         position. Each team has its own channel and the value represents the number
#         of agents of that team occupying the same square.
#         """
#         my_agent = self.agents[my_id]
#         if isinstance(my_agent, ObservingAgent) and isinstance(my_agent, TeamAgent):
#             signal = np.zeros((my_agent.view*2+1, my_agent.view*2+1))

#             # --- Determine the boundaries of the agents' grids --- #
#             # For left and top, we just do: view - x,y >= 0
#             # For the right and bottom, we just do region - x,y - 1 - view > 0
#             if my_agent.view - my_agent.position[0] >= 0: # Top end
#                 signal[0:my_agent.view - my_agent.position[0], :] = -1
#             if my_agent.view - my_agent.position[1] >= 0: # Left end
#                 signal[:, 0:my_agent.view - my_agent.position[1]] = -1
#             if self.region - my_agent.position[0] - my_agent.view - 1 < 0: # Bottom end
#                 signal[self.region - my_agent.position[0] - my_agent.view - 1:,:] = -1
#             if self.region - my_agent.position[1] - my_agent.view - 1 < 0: # Right end
#                 signal[:, self.region - my_agent.position[1] - my_agent.view - 1:] = -1

#             # Repeat the boundaries signal for all teams
#             signal = np.repeat(signal[:, :, np.newaxis], self.number_of_teams, axis=2)

#             # --- Determine the positions of all the other alive agents --- #
#             for other_id, other_agent in self.agents.items():
#                 if other_id == my_id: continue # Don't observe yourself
#                 r_diff = other_agent.position[0] - my_agent.position[0]
#                 c_diff = other_agent.position[1] - my_agent.position[1]
#                 if -my_agent.view <= r_diff <= my_agent.view and -my_agent.view <= c_diff <= my_agent.view:
#                     r_diff += my_agent.view
#                     c_diff += my_agent.view
#                     signal[r_diff, c_diff, other_agent.team] += 1

#             return signal
