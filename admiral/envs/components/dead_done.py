
import numpy as np

from admiral.envs.components.agent import LifeAgent
from admiral.envs.components.agent import TeamAgent

class DeadDone:
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
    
    def get_done(self, agent, **kwargs):
        """
        Return True if the agent is dead. Otherwise, return False.
        """
        return not agent.is_alive

    def get_all_done(self, **kwargs):
        """
        Return True if all agents are dead. Otherwise, return False.
        """
        for agent in self.agents.values():
            if agent.is_alive:
                return False
        return True

class TeamDeadDone:
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
    
    def get_done(self, agent, **kwargs):
        """
        Return True if the agent is dead. Otherwise, return False.
        """
        return not agent.is_alive

    def get_all_done(self, **kwargs):
        """
        Return true if the only agent left alive are all on the same team. Otherwise,
        return false.
        """
        team = np.zeros(self.number_of_teams)
        for agent in self.agents.values():
            if agent.is_alive:
                team[agent.team] += 1
        return sum(team != 0) <= 1
