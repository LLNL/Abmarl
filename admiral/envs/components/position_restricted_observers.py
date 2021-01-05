

from admiral.envs.components.observers import AgentObservingAgent # TODO: probably change this name...
from admiral.envs.components.position import PositionAgent, PositionObserver
from admiral.envs.components.team import TeamObserver
from admiral.envs.components.health import HealthObserver, LifeObserver

def obs_filter(obs, agent, agents, null_value):
    if isinstance(agent, PositionAgent) and \
        isinstance(agent, AgentObservingAgent):
        for other in agents.values():
            if other.id == agent.id: continue # Don't modify yourself
            if not isinstance(other, PositionAgent): continue # Can't modify based on position
            r_diff = abs(other.position[0] - agent.position[0])
            c_diff = abs(other.position[1] - agent.position[1])
            if r_diff > agent.agent_view or c_diff > agent.agent_view: # Agent too far away to observe
                obs[other.id] = null_value
    return obs

class PositionRestrictedTeamObserver(TeamObserver):
    """
    Observe the team of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        return obs_filter(super().get_obs(), agent, self.agents, self.null_value)

class PositionRestrictedPositionObserver(PositionObserver):
    """
    Observe the position of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        return obs_filter(super().get_obs(), agent, self.agents, self.null_value)

class PositionRestrictedRelativePositionObserver(RelativePositionObserver):
    """
    Observe the relative position of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        return obs_filter(super().get_obs(), agent, self.agents, self.null_value)

class PositionRestrictedHealthObserver(HealthObserver):
    """
    Observe the health of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        return obs_filter(super().get_obs(), agent, self.agents, self.null_value)

class PositionRestrictedLifeObserver(LifeObserver):
    """
    Observe the life of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        return obs_filter(super().get_obs(), agent, self.agents, self.null_value)
