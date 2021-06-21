
from abc import ABC, abstractmethod

import numpy as np

from abmarl.sim.gridworld.base import GridWorldBaseComponent


class StateBaseComponent(GridWorldBaseComponent, ABC):
    """
    Abstract State Component base from which all state components will inherit.

    In addition to the Grid World Components, every State Component has access
    to the grid.
    """
    def __init__(self, grid=None, **kwargs):
        self.grid = grid
        self._rows = self.grid.shape[0]
        self._cols = self.grid.shape[1]

    @property
    def rows(self):
        """
        The number of rows in the grid.
        """
        return self._rows

    @property
    def cols(self):
        """
        The number of columns in the grid.
        """
        return self._cols

    @property
    def grid(self):
        """
        The grid indexes the agents by their position.

        For example, an agent whose position is (3, 2) can be accessed through
        the grid with self.grid[3, 2]. Components are responsible for maintaining
        the connection between agent position and grid index.
        """
        return self._grid

    @grid.setter
    def grid(self, value):
        assert type(value) is np.ndarray, "The grid must be a numpy array."
        assert len(value.shape) == 2, "The grid must be a 2-dimensional array."
        assert value.dtype is np.dtype(object), "The grid must be a numpy array of objects."
        self._grid = value

    @abstractmethod
    def reset(self, **kwargs):
        """
        Resets the part of the state for which it is responsible.
        """
        pass


class PositionState(StateBaseComponent):
    """
    Manage the agent's positions in the grid.

    Every agent occupies a unique cell.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        We use the agent's initial position if it exists. Otherwise, we randomly
        place the agent in the grid.
        """
        # Grid lookup by position
        self.grid.fill(None)
        # Prioritize placing agents with initial positions. We must keep track
        # of which positions have been taken so that the random placement below doesn't
        # try to place an agent in an already-taken position.
        ravelled_positions_taken = set()
        for agent in self.agents.values():
            if agent.initial_position is not None:
                r, c = agent.initial_position
                assert self.grid[r, c] is None, f"{agent.id} has the same initial " + \
                    f"position as {self.grid[r, c].id}. All initial positions must be unique."
                agent.position = agent.initial_position
                self.grid[r, c] = agent
                ravelled_positions_taken.add(
                    np.ravel_multi_index(agent.position, (self.rows, self.cols))
                )

        # Now randomly place any other agent who did not come with an initial position.
        ravelled_positions_available = [
            i for i in range(self.rows * self.cols) if i not in ravelled_positions_taken
        ]
        rs, cs = np.unravel_index(
            np.random.choice(ravelled_positions_available, len(self.agents), False),
            shape=(self.rows, self.cols)
        )
        for ndx, agent in enumerate(self.agents.values()):
            if agent.initial_position is None:
                r = rs[ndx]
                c = cs[ndx]
                agent.position = np.array([r, c])
                self.grid[r, c] = agent
