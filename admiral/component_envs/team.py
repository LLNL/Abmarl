
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
