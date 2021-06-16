
from gym.spaces import Box
import numpy as np

from abmarl.sim import PrincipleAgent, Agent, ActingAgent, ObservingAgent, AgentBasedSimulation


class GridWorldAgent(PrincipleAgent):
    """
    The basic entity in the GridWorld.
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
            assert value.shape == (2,), "Initial position must be a 2-dimensional array."
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
        return self._view_blocking
    
    @view_blocking.setter
    def view_blocking(self, value):
        assert type(value) is bool, "View blocking must be either True or False."
        self._view_blocking = value


class GridObservingAgent(GridWorldAgent, ObservingAgent):
    def __init__(self, view_range=None, **kwargs):
        super().__init__(**kwargs)
        self.view_range = view_range
        self.observation_space['grid'] = Box(-np.inf, np.inf, (view_range, view_range), np.int)


class MovingAgent(GridWorldAgent, ActingAgent):
    def __init__(self, move_range=None, **kwargs):
        super().__init__(**kwargs)
        self.move_range = move_range
        self.action_space['move'] = Box(-move_range, move_range, (2,), np.int)



