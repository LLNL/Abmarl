
from admiral.envs import Agent

class ObservingAgent(Agent):
    def __init__(self, view=None, **kwargs):
        assert view is not None, "view must be nonnegative integer"
        self.view = view
        super().__init__(**kwargs)

    @property
    def configured(self):
        return super().configured and self.view is not None
