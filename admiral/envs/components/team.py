
import numpy as np

from admiral.envs import Agent

class TeamAgent(Agent):
    """
    Agents are on a team, which will affect their ability to perform certain actions,
    such as who they can attack.
    """
    def __init__(self, team=None, **kwargs):
        super().__init__(**kwargs)
        assert team is not None, "team must be an integer"
        self.team = team
    
    @property
    def configured(self):
        """
        Agent is configured if team is set.
        """
        return super().configured and self.team is not None

class TeamComponent:
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        self.number_of_teams = number_of_teams
        self.agents = agents

class TeamObserver:
    def __init__(self, team=None, agents=None, **kwargs):
        self.team = team
        self.agents = agents
    
        from gym.spaces import Box, Dict
        for agent in agents.values():
            agent.observation_space['team'] = Dict({
                agent_id: Box(0, self.team.number_of_teams, (1,), np.int) for agent_id in agents
            })
    
    def get_obs(self, *args, **kwargs):
        return {other.id: self.agents[other.id].team for other in self.agents.values()}
