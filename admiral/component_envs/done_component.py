
import numpy as np

from admiral.component_envs.death_life import LifeAgent
from admiral.component_envs.team import TeamAgent

class DeadDoneComponent:
    def __init__(self, agents=None, **kwargs):
        for agent in agents.values():
            assert isinstance(agent, LifeAgent)
        self.agents = agents
    
    def get_done(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return False if agent.is_alive else True

    def get_all_done(self, **kwargs):
        for agent in self.agents.values():
            if agent.is_alive:
                return False
        return True

class TeamDeadDoneComponent:
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        for agent in agents.values():
            assert isinstance(agent, TeamAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents
        assert type(number_of_teams) is int, "number_of_teams must be a positive integer."
        self.number_of_teams = number_of_teams
    
    def get_done(self, agent_id, **kwargs):
        return False if self.agents[agent_id].is_alive else True

    def get_all_done(self, **kwargs):
        team = np.zeros(self.number_of_teams)
        for agent in self.agents.values():
            if agent.is_alive:
                team[agent.team] += 1
        return False if sum(team != 0) > 1 else True
