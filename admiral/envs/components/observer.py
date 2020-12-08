
from admiral.envs import Agent

class ObservingAgent(Agent):
    """
    Agents can observe features of the environment up to a certain view distance
    away.

    view (int):
        The view distance.
    """
    def __init__(self, view=None, **kwargs):
        super().__init__(**kwargs)
        assert view is not None, "view must be nonnegative integer"
        self.view = view

    @property
    def configured(self):
        """
        Agents are configured if the view is set.
        """
        return super().configured and self.view is not None
