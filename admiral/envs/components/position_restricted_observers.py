
from gym.spaces import Dict, MultiBinary

from admiral.envs import Agent
from admiral.envs.components.agent import AgentObservingAgent
from admiral.envs.components.position import PositionAgent, PositionObserver, RelativePositionObserver
from admiral.envs.components.team import TeamObserver
from admiral.envs.components.health import HealthObserver, LifeObserver


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

class PositionRestrictedTeamObserver(TeamObserver):
    """
    Observe the team of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs()
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}

class PositionRestrictedPositionObserver(PositionObserver):
    """
    Observe the position of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}

class PositionRestrictedRelativePositionObserver(RelativePositionObserver):
    """
    Observe the relative position of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}

class PositionRestrictedHealthObserver(HealthObserver):
    """
    Observe the health of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs()
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}

class PositionRestrictedLifeObserver(LifeObserver):
    """
    Observe the life of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs()
        obs_key = next(iter(obs))
        return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
