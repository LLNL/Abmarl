
import numpy as np

from admiral.component_envs.death_life import DyingAgent
from admiral.component_envs.team import TeamAgent

class DoneConditioner:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        self.dones = {agent_id: False for agent_id in self.agents}

    def get_done(self, agent_id, **kwargs):
        return self.dones[agent_id]
    
    def get_all_done(self, **kwargs):
        return all([self.get_done(agent_id) for agent_id in self.agents])

class DeadDoneComponent:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
    
    def get_done(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        if isinstance(agent, DyingAgent):
            return False if agent.is_alive else True

    def get_all_done(self, **kwargs):
        for agent in self.agents.values():
            if isinstance(agent, DyingAgent):
                if agent.is_alive:
                    return False
        return True

class TeamDeadDoneComponent:
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        for agent in agents.values():
            assert isinstance(agent, TeamAgent) and isinstance(agent, DyingAgent)
        self.agents = agents
        self.number_of_teams = number_of_teams
    
    def get_done(self, agent_id, **kwargs):
        return False if self.agents[agent_id].is_alive else True

    def get_all_done(self, **kwargs):
        team = np.zeros(self.number_of_teams)
        for agent in self.agents.values():
            if agent.is_alive:
                team[agent.team-1] += 1
        return False if sum(team != 0) > 1 else True
