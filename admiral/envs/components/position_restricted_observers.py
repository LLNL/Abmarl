

from admiral.envs.components.observers import AgentObservingAgent
from admiral.envs.components.position import PositionAgent
from admiral.envs.components.team import TeamAgent, TeamObserver

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


