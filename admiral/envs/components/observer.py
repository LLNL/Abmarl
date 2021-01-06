
from gym.spaces import Box, MultiBinary, Dict
import numpy as np

from admiral.envs.components.agent import LifeAgent, PositionAgent, AgentObservingAgent, TeamAgent, ResourceObservingAgent

# ----------------- #
# --- Utilities --- #
# ----------------- #

def obs_filter(obs, agent, agents, null_value):
    if isinstance(agent, PositionAgent) and \
       isinstance(agent, AgentObservingAgent):
        for other in agents.values():
            if other.id not in obs: continue # This agent is not observed
            if other.id == agent.id: continue # Don't modify yourself
            if not isinstance(other, PositionAgent):
                # Agents without positions will not be observed at all
                # TODO: make this a parameter to the observers below
                obs[other.id] = null_value
            else:
                r_diff = abs(other.position[0] - agent.position[0])
                c_diff = abs(other.position[1] - agent.position[1])
                if r_diff > agent.agent_view or c_diff > agent.agent_view: # Agent too far away to observe
                    obs[other.id] = null_value
    return obs



# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

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

class PositionRestrictedHealthObserver(HealthObserver):
    """
    Observe the health of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs()
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}


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

class PositionRestrictedLifeObserver(LifeObserver):
    """
    Observe the life of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs()
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}



# --------------- #
# --- Masking --- #
# --------------- #

class MaskObserver:
    # TODO: Having a single mask observer may actually be confusing because it will
    # have every single agent in the dictionary, whereas the actual observer may
    # only have a subset of them. For example, a life observer will only include
    # life agents, and if this mask is applied, then the mask will include all
    # the agents, so there is not a clear mapping to the actual observation.
    #
    # Paths to take:
    # (1) Change the observers so that all agents are always observed
    # and the ones that are not life agents will simply be observed with the null
    # value.
    # (2) Have a separate mask for every observer as part of the restricted classes
    # below.
    # We should look at some rllib examples to help us decide the right way forward.
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent):
                agent.observation_space['mask'] = Dict({
                    other: MultiBinary(1) for other in agents
                })
    
    def get_obs(self, agent, **kwargs):
        if isinstance(agent, AgentObservingAgent):
            return {'mask': {other: True for other in self.agents}}
        else:
            return {}
    
    @property
    def null_value(self):
        return False

class PositionRestrictedMaskObserver(MaskObserver):
    """
    Set the mask value to False for any agent that is too far way from the observing
    agent.
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}



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
        from gym.spaces import Dict, Box
        for agent in agents.values():
            agent.observation_space['position'] = Dict({
                other.id: Box(-1, self.position.region, (2,), np.int) for other in agents.values() if isinstance(other, PositionAgent)
            })

    def get_obs(self, agent, **kwargs):
        """
        Get the positions of all the agents in the simulator.
        """
        return {'position': {other.id: other.position for other in self.agents.values() if isinstance(other, PositionAgent)}}
    
    @property
    def null_value(self):
        return np.array([-1, -1])

class PositionRestrictedPositionObserver(PositionObserver):
    """
    Observe the position of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}


class RelativePositionObserver:
    """
    Observe the relative positions of agents in the simulator.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, PositionAgent):
                agent.observation_space['position'] = Dict({
                    other.id: Box(-position.region, position.region, (2,), np.int) for other in agents.values() if (other.id != agent.id and isinstance(other, PositionAgent))
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the relative positions of all the agents in the simulator.
        """
        if isinstance(agent, PositionAgent):
            obs = {}
            for other in self.agents.values():
                if other.id == agent.id: continue # Don't observe your own position
                if not isinstance(other, PositionAgent): continue # Can't observe relative position from agents who do not have a position.
                r_diff = other.position[0] - agent.position[0]
                c_diff = other.position[1] - agent.position[1]
                obs[other.id] = np.array([r_diff, c_diff])
            return {'position': obs}
        else:
            return {}
    
    @property
    def null_value(self):
        return np.array([-self.position.region, -self.position.region])

class PositionRestrictedRelativePositionObserver(RelativePositionObserver):
    """
    Observe the relative position of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}


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
        self.agents = agents
        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionAgent):
                agent.observation_space['position'] = Box(-1, 1, (agent.agent_view*2+1, agent.agent_view*2+1), np.int)

    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position.
        """
        if isinstance(my_agent, AgentObservingAgent) and isinstance(my_agent, PositionAgent):
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
                if not isinstance(other_agent, PositionAgent): continue # Can only observer position of PositionAgents
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
            if isinstance(agent, AgentObservingAgent) and isinstance(agent, PositionAgent):
                agent.observation_space['position'] = Box(-1, np.inf, (agent.agent_view*2+1, agent.agent_view*2+1, self.team_state.number_of_teams), np.int)
    
    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position. Each team has its own channel and the value represents the number
        of agents of that team occupying the same square.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, TeamAgent) and \
           isinstance(my_agent, PositionAgent):
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
            signal = np.repeat(signal[:, :, np.newaxis], self.team_state.number_of_teams, axis=2)

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not isinstance(other_agent, PositionAgent): continue # Cannot observe agent without position
                if not isinstance(other_agent, TeamAgent): continue # Cannot observe agent without team.
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff, other_agent.team] += 1

            return {'position': signal}
        else:
            return {}



# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class GridResourceObserver:
    """
    Agents observe a grid of size resource_view_range centered on their
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

        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, ResourceObservingAgent):
                agent.observation_space['resources'] = Box(0, self.resources.max_value, (agent.resource_view_range*2+1, agent.resource_view_range*2+1), np.float)

    def get_obs(self, agent, **kwargs):
        """
        These cells are filled with the values of the resources surrounding the
        agent's position.
        """
        if isinstance(agent, ResourceObservingAgent):
            signal = -np.ones((agent.resource_view_range*2+1, agent.resource_view_range*2+1))

            # Derived by considering each square in the resources as an "agent" and
            # then applied the agent diff logic from above. The resulting for-loop
            # can be written in the below vectorized form.
            (r,c) = agent.position
            r_lower = max([0, r-agent.resource_view_range])
            r_upper = min([self.resources.region-1, r+agent.resource_view_range])+1
            c_lower = max([0, c-agent.resource_view_range])
            c_upper = min([self.resources.region-1, c+agent.resource_view_range])+1
            signal[(r_lower+agent.resource_view_range-r):(r_upper+agent.resource_view_range-r),(c_lower+agent.resource_view_range-c):(c_upper+agent.resource_view_range-c)] = \
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

class PositionRestrictedTeamObserver(TeamObserver):
    """
    Observe the team of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs()
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
