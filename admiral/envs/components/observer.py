
from gym.spaces import Box, Discrete, Dict
import numpy as np

from admiral.envs.components.agent import HealthObservingAgent, LifeObservingAgent, \
    AgentObservingAgent, PositionObservingAgent, SpeedAngleObservingAgent, VelocityObservingAgent, \
    ResourceObservingAgent, TeamObservingAgent, BroadcastObservingAgent, PositionAgent, LifeAgent, \
    TeamAgent, SpeedAngleAgent, VelocityAgent, BroadcastingAgent

# --------------------- #
# --- Communication --- #
# --------------------- #

class BroadcastObserver:
    """
    Observe the broadcast state of broadcasting agents.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        for agent in agents.values():
            if isinstance(agent, BroadcastObservingAgent):
                agent.observation_space[self.channel] = Dict({
                    other: Box(-1, 1, (1,)) for other in self.agents.values()
                })
        
    def get_obs(self, agent, **kwargs):
        if isinstance(agent, BroadcastObservingAgent):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, BroadcastingAgent):
                    obs[other.id] = other.broadcasting
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}
    
    @property
    def channel(self):
        return 'broadcast'
    
    @property
    def null_value(self):
        return -1

# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class HealthObserver:
    """
    Observe the health state of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        for agent in agents.values():
            if isinstance(agent, HealthObservingAgent):
                obs_space = {}
                for other in self.agents.values():
                    if isinstance(other, LifeAgent):
                        obs_space[other.id] = Box(-1, other.max_health, (1,))
                    else:
                        obs_space[other.id] = Box(-1, -1, (1,))
                agent.observation_space[self.channel] = Dict(obs_space)
    
    def get_obs(self, agent, **kwargs):
        """
        Get the health state of all the agents in the simulator.

        agent (HealthObservingAgent):
            The agent making the observation.
        """
        if isinstance(agent, HealthObservingAgent):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, LifeAgent):
                    obs[other.id] = other.health
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}
    
    @property
    def channel(self):
        return 'health'
    
    @property
    def null_value(self):
        return -1


class LifeObserver:
    """
    Observe the life state of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        for agent in agents.values():
            if isinstance(agent, LifeObservingAgent):
                agent.observation_space[self.channel] = Dict({
                    other.id: Box(-1, 1, (1,), np.int) for other in self.agents.values()
                })
    
    def get_obs(self, agent, **kwargs):
        """
        Get the life state of all the agents in the simulator.

        agent (LifeObservingAgent):
            The agent making the observation.
        """
        if isinstance(agent, LifeObservingAgent):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, LifeAgent):
                    obs[other.id] = np.array([other.is_alive])
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}
    
    @property
    def channel(self):
        return 'life'
    
    @property
    def null_value(self):
        return np.array([-1])



# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class PositionObserver:
    """
    Observe the positions of all the agents in the simulator.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        
        for agent in agents.values():
            if isinstance(agent, PositionObservingAgent):
                agent.observation_space[self.channel] = Dict({
                    other.id: Box(-1, self.position.region, (2,), np.int) for other in agents.values()
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the positions of all the agents in the simulator.
        """
        if isinstance(agent, PositionObservingAgent):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, PositionAgent):
                    obs[other.id] = other.position
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}   
    
    @property
    def channel(self):
        return 'position'
    
    @property
    def null_value(self):
        return np.array([-1, -1])


class RelativePositionObserver:
    """
    Observe the relative positions of agents in the simulator.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        
        for agent in agents.values():
            if isinstance(agent, PositionObservingAgent) and \
               isinstance(agent, PositionAgent):
                agent.observation_space[self.channel] = Dict({
                    other.id: Box(-position.region, position.region, (2,), np.int) for other in agents.values()
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the relative positions of all the agents in the simulator.
        """
        if isinstance(agent, PositionObservingAgent) and \
           isinstance(agent, PositionAgent):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, PositionAgent):
                    r_diff = other.position[0] - agent.position[0]
                    c_diff = other.position[1] - agent.position[1]
                    obs[other.id] = np.array([r_diff, c_diff])
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}
    
    @property
    def channel(self):
        return 'relative_position'
    
    @property
    def null_value(self):
        return np.array([-self.position.region, -self.position.region])


class GridPositionBasedObserver:
    """
    Agents observe a grid of size agent_view centered on their
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
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
        self.agents = agents
        
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionAgent) and \
               isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Box(-1, 1, (agent.agent_view*2+1, agent.agent_view*2+1), np.int)

    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, PositionAgent) and \
           isinstance(my_agent, PositionObservingAgent):
            signal = np.zeros((my_agent.agent_view*2+1, my_agent.agent_view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.agent_view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.agent_view - my_agent.position[0], :] = -1
            if my_agent.agent_view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.agent_view - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.agent_view - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.agent_view - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.agent_view - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.agent_view - 1:] = -1

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not isinstance(other_agent, PositionAgent): continue # Can only observe position of PositionAgents
                if not (isinstance(other_agent, LifeAgent) and other_agent.is_alive): continue # Can only observe alive agents
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff] = 1 # There is an agent at this location.

            return {'position': signal}
        else:
            return {}

class GridPositionTeamBasedObserver:
    """
    Agents observe a grid of size agent_view centered on their
    position. The observation contains one channel per team, where the value of
    the cell is the number of agents on that team that occupy that square. -1
    indicates out of bounds.
    
    position (PositionState):
        The position state handler, which contains the region.

    number_of_teams (int):
        The number of teams in this simuation.
        Default 0.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, number_of_teams=0, agents=None, **kwargs):
        self.position = position
        self.number_of_teams = number_of_teams + 1
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, TeamAgent)
        self.agents = agents

        for agent in self.agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionAgent) and \
               isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Box(-1, np.inf, (agent.agent_view*2+1, agent.agent_view*2+1, self.number_of_teams), np.int)
    
    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position. Each team has its own channel and the value represents the number
        of agents of that team occupying the same square.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, TeamAgent) and \
           isinstance(my_agent, PositionAgent) and \
           isinstance(my_agent, PositionObservingAgent):
            signal = np.zeros((my_agent.agent_view*2+1, my_agent.agent_view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.agent_view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.agent_view - my_agent.position[0], :] = -1
            if my_agent.agent_view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.agent_view - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.agent_view - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.agent_view - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.agent_view - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.agent_view - 1:] = -1

            # Repeat the boundaries signal for all teams
            signal = np.repeat(signal[:, :, np.newaxis], self.number_of_teams, axis=2)

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not isinstance(other_agent, PositionAgent): continue # Cannot observe agent without position
                if not isinstance(other_agent, TeamAgent): continue # Cannot observe agent without team.
                if not (isinstance(other_agent, LifeAgent) and other_agent.is_alive): continue # Can only observe alive agents
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff, other_agent.team] += 1

            return {'position': signal}
        else:
            return {}

class SpeedObserver:
    """
    Observe the speed of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        
        for agent in agents.values():
            if isinstance(agent, SpeedAngleObservingAgent):
                obs_space = {}
                for other in self.agents.values():
                    if isinstance(other, SpeedAngleAgent):
                        obs_space[other.id] = Box(-1, other.max_speed, (1,))
                    else:
                        obs_space[other.id] = Box(-1, -1, (1,))
                agent.observation_space[self.channel] = Dict(obs_space)

    def get_obs(self, agent, **kwargs):
        """
        Get the speed of all the agents in the simulator.
        """
        if isinstance(agent, SpeedAngleObservingAgent):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, SpeedAngleAgent):
                    obs[other.id] = other.speed
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}
    
    @property
    def channel(self):
        return 'speed'
    
    @property
    def null_value(self):
        return -1


class AngleObserver:
    """
    Observe the angle of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        for agent in agents.values():
            if isinstance(agent, SpeedAngleObservingAgent):
                agent.observation_space[self.channel] = Dict({
                    other.id: Box(-1, 360, (1,)) for other in agents.values()
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the angle of all the agents in the simulator.
        """
        if isinstance(agent, SpeedAngleObservingAgent):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, SpeedAngleAgent):
                    obs[other.id] = other.ground_angle
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}
    
    @property
    def channel(self):
        return 'ground_angle'
    
    @property
    def null_value(self):
        return -1


class VelocityObserver:
    """
    Observe the velocity of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        
        for agent in agents.values():
            if isinstance(agent, VelocityObservingAgent):
                obs_space = {}
                for other in self.agents.values():
                    if isinstance(other, VelocityAgent):
                        obs_space[other.id] = Box(-agent.max_speed, agent.max_speed, (2,))
                    else:
                        obs_space[other.id] = Box(0, 0, (2,))
                agent.observation_space[self.channel] = Dict(obs_space)
    
    def get_obs(self, agent, **kwargs):
        """
        Get the velocity of all the agents in the simulator.
        """
        if isinstance(agent, VelocityObservingAgent):
            obs = {}
            for other in self.agents.values():
                if isinstance(other, VelocityAgent):
                    obs[other.id] = other.velocity
                else:
                    obs[other.id] = self.null_value
            return {self.channel: obs}
        else:
            return {}
    
    @property
    def channel(self):
        return 'velocity'
    
    @property
    def null_value(self):
        return np.zeros(2)



# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class GridResourceObserver:
    """
    Agents observe a grid of size resource_view centered on their
    position. The values in the grid are the values of the resources in that
    area.

    resources (ResourceState):
        The resource state handler.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, resources=None, agents=None, **kwargs):
        self.resources = resources
        self.agents = agents

        for agent in agents.values():
            if isinstance(agent, ResourceObservingAgent):
                agent.observation_space['resources'] = Box(-1, self.resources.max_value, (agent.resource_view*2+1, agent.resource_view*2+1))

    def get_obs(self, agent, **kwargs):
        """
        These cells are filled with the values of the resources surrounding the
        agent's position.
        """
        if isinstance(agent, ResourceObservingAgent):
            signal = -np.ones((agent.resource_view*2+1, agent.resource_view*2+1))

            # Derived by considering each square in the resources as an "agent" and
            # then applied the agent diff logic from above. The resulting for-loop
            # can be written in the below vectorized form.
            (r,c) = agent.position.astype(int)
            r_lower = max([0, r-agent.resource_view])
            r_upper = min([self.resources.region-1, r+agent.resource_view])+1
            c_lower = max([0, c-agent.resource_view])
            c_upper = min([self.resources.region-1, c+agent.resource_view])+1
            signal[(r_lower+agent.resource_view-r):(r_upper+agent.resource_view-r),(c_lower+agent.resource_view-c):(c_upper+agent.resource_view-c)] = \
                self.resources.resources[r_lower:r_upper, c_lower:c_upper]
            return {'resources': signal}
        else:
            return {}



# ------------ #
# --- Team --- #
# ------------ #

class TeamObserver:
    """
    Observe the team of each agent in the simulator.
    """
    def __init__(self, number_of_teams=0, agents=None, **kwargs):
        self.number_of_teams = number_of_teams
        self.agents = agents
    
        for agent in agents.values():
            if isinstance(agent, TeamObservingAgent):
                agent.observation_space[self.channel] = Dict({
                    other.id: Box(-1, self.number_of_teams, (1,), np.int) for other in agents.values()
                })
    
    def get_obs(self, agent, **kwargs):
        """
        Get the team of each agent in the simulator.
        """
        if isinstance(agent, TeamObservingAgent):
            return {'team': {other.id: np.array([other.team]) for other in self.agents.values()}}
        else:
            return {}
    
    @property
    def channel(self):
        return 'team'
    
    @property
    def null_value(self):
        return np.array([-1])
