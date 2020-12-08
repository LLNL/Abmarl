
import numpy as np

from admiral.component_envs.death_life import LifeAgent
from admiral.component_envs.team import TeamAgent

class DeadDoneComponent:
    """
    Dead agents are indicated as done. Additionally, the simulation is over when
    all the agents are dead.

    agents (dict):
        The dictionary of agents. Because the done condition is determined by
        the agent's life status, all agents must be LifeAgents.
    """
    def __init__(self, agents=None, **kwargs):
        for agent in agents.values():
            assert isinstance(agent, LifeAgent)
        self.agents = agents
    
    def get_done(self, agent_id, **kwargs):
        """
        Return true if the agent is dead. Otherwise, return false.
        """
        agent = self.agents[agent_id]
        return False if agent.is_alive else True

    def get_all_done(self, **kwargs):
        """
        Return true if all agents are dead. Otherwise, return false.
        """
        for agent in self.agents.values():
            if agent.is_alive:
                return False
        return True

class TeamDeadDoneComponent:
    """
    Dead agents are indicated as done. Additionally, the simulation is over when
    the only agents remaining are on the same team.

    agents (dict):
        The dictionary of agents. Because the done condition is determined by the
        agent's life status, all agents must be LifeAgents; and because the done
        condition is determined by the agents' teams, all agents must be TeamAgents.

    number_of_teams (int):
        The fixed number of teams in this simulation.
    """
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        for agent in agents.values():
            assert isinstance(agent, TeamAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents
        assert type(number_of_teams) is int, "number_of_teams must be a positive integer."
        self.number_of_teams = number_of_teams
    
    def get_done(self, agent_id, **kwargs):
        """
        Return true if the agent is dead. Otherwise, return false.
        """
        return False if self.agents[agent_id].is_alive else True

    def get_all_done(self, **kwargs):
        """
        Return true if the only agent left alive are all on the same team. Otherwise,
        return false.
        """
        team = np.zeros(self.number_of_teams)
        for agent in self.agents.values():
            if agent.is_alive:
                team[agent.team] += 1
        return False if sum(team != 0) > 1 else True
