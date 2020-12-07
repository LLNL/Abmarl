
from admiral.envs import Agent

class ObservingAgent(Agent):
    def __init__(self, view=None, **kwargs):
        super().__init__(**kwargs)
        assert view is not None, "view must be nonnegative integer"
        self.view = view

    @property
    def configured(self):
        return super().configured and self.view is not None
