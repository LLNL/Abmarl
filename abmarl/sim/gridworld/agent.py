
import numpy as np

from abmarl.sim import PrincipleAgent, ActingAgent, ObservingAgent


class GridWorldAgent(PrincipleAgent):
    """
    The base agent in the GridWorld.
    """
    def __init__(self, initial_position=None, view_blocking=False, **kwargs):
        super().__init__(**kwargs)
        self.initial_position = initial_position
        self.view_blocking = view_blocking

    @property
    def initial_position(self):
        """
        The agent's initial position at reset.
        """
        return self._initial_position

    @initial_position.setter
    def initial_position(self, value):
        if value is not None:
            assert type(value) is np.ndarray, "Initial position must be a numpy array."
            assert value.shape == (2,), "Initial position must be a 2-element array."
            assert value.dtype in [np.int, np.float], "Initial position must be numerical."
        self._initial_position = value

    @property
    def position(self):
        """
        The agent's position in the grid.
        """
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def encoding(self):
        """
        The numerical value given to other agents who observe this agent.
        """
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        assert type(value) is int, f"{self.id}'s encoding must be an integer."
        assert value != -2, "-2 encoding reserved for masked observation."
        assert value != -1, "-1 encoding reserved for out of bounds."
        assert value != 0, "0 encoding reserved for empty cell."
        self._encoding = value

    @property
    def render_shape(self):
        """
        The agent's shape in the rendered grid.
        """
        return getattr(self, '_render_shape', 's')

    @render_shape.setter
    def render_shape(self, value):
        self._render_shape = value

    @property
    def view_blocking(self):
        """
        Specify if this agent blocks other agent's observations.
        """
        return self._view_blocking

    @view_blocking.setter
    def view_blocking(self, value):
        assert type(value) is bool, "View blocking must be either True or False."
        self._view_blocking = value


class GridObservingAgent(GridWorldAgent, ObservingAgent):
    """
    Observe the grid up to view_range cells.

    Attributes:
        view_range: The number of cells this agent can observe in each step.
    """
    def __init__(self, view_range=None, **kwargs):
        super().__init__(**kwargs)
        self.view_range = view_range


class MovingAgent(GridWorldAgent, ActingAgent):
    """
    Move up to move_range cells.

    Attributes:
        move_range: The number of cells this ageant can move in one step.
    """
    def __init__(self, move_range=None, **kwargs):
        super().__init__(**kwargs)
        self.move_range = move_range


class HealthAgent(GridWorldAgent):
    """
    Agents have health points and can die.

    Health is bounded between 0 and 1.

    Attributes:
        initial_health: The agent's initial health between 0 and 1.
    """
    def __init__(self, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_health = initial_health
    
    @property
    def initial_health(self):
        return self._initial_health
    
    @initial_health.setter
    def initial_health(self, value):
        assert type(value) in [int, float], "Initial health must be a numeric value."
        assert 0 < value <= 1, "Initial value must be between 0 and 1."
        self._initial_health = value
