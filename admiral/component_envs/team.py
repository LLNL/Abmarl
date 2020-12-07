
from admiral.envs import Agent

class TeamAgent(Agent):
    def __init__(self, team=None, **kwargs):
        super().__init__(**kwargs)
        assert team is not None, "team must be an integer"
        self.team = team
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return super().configured and self.team is not None
