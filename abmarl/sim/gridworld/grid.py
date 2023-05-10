
import copy

import numpy as np


class Grid:
    """
    A Grid stores the agents at indices in a numpy array.

    Components can interface with the Grid. Each index in the
    grid is a dictionary that maps the agent id to the agent object itself. If agents
    can overlap, then there may be more than one agent per cell.

    Args:
        rows: The number of rows in the grid.
        cols: The number of columns in the grid.
        overlapping: Overlapping matrix tracks which agents can overlap based on their
            encodings.
    """
    def __init__(self, rows, cols, overlapping=None, **kwargs):
        assert type(rows) is int and rows > 0, "Rows must be a positive integer."
        assert type(cols) is int and cols > 0, "Cols must be a positive integer."
        self._internal = np.empty((rows, cols), dtype=object)
        self.overlapping = overlapping

    @property
    def rows(self):
        """
        The number of rows in the grid.
        """
        return self._internal.shape[0]

    @property
    def cols(self):
        """
        The number of columns in the grid.
        """
        return self._internal.shape[1]

    @property
    def overlapping(self):
        """
        Overlapping matrix tracks which agents can overlap based on their encodings.

        A dictionary that maps agents' encodings to a set of encodings with which
        they can overlap. If the overlapping matrix is not symmetrical,
        then we update it here to be symmetrical. That is, if 2 can overlap with
        3, then 3 can overlap with 2.
        """
        return self._overlapping

    @overlapping.setter
    def overlapping(self, value):
        if value is not None:
            assert type(value) is dict, "Overlaping must be dictionary."
            symmetric_value = copy.deepcopy(value)
            for ndx, overlap_set in value.items():
                assert type(ndx) is int, "All keys in overlapping dict must be integers."
                assert type(overlap_set) is set, "All values in overlapping dict must be sets."
                for overlap_ndx in overlap_set:
                    assert type(overlap_ndx) is int, \
                        "All elements in overlapping dict values must be integers."
                    # Force symmetry in the overlapping
                    if overlap_ndx not in symmetric_value:
                        symmetric_value[overlap_ndx] = {ndx}
                    else:
                        symmetric_value[overlap_ndx].add(ndx)
            self._overlapping = symmetric_value
        else:
            self._overlapping = {}

    def reset(self, **kwargs):
        """
        Reset the grid to an empty state.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self._internal[i, j] = {}

    def query(self, agent, ndx):
        """
        Query a cell in the grid to see if is available to this agent.

        The cell is available for the agent if it is empty or if both the occupying agent
        and the querying agent are overlappable.

        Args:
            agent: The agent for which we are checking availabilty.
            ndx: The cell to query.

        Returns:
            The availability of this cell.
        """
        ndx = tuple(ndx)
        if self._internal[ndx]: # There are agents here
            try:
                return all([
                    True if other.encoding in self._overlapping[agent.encoding] else False
                    for other in self._internal[ndx].values()
                ])
            except KeyError:
                return False
        else:
            return True

    def place(self, agent, ndx):
        """
        Place an agent at an index.

        If the cell is available, the agent will be placed at that index
        in the grid and the agent's position will be updated. The placement is
        successful if the new position is unoccupied or if the agent already occupying
        that position is overlappable AND this agent is overlappable.

        Args:
            agent: The agent to place.
            ndx: The new index for this agent.

        Returns:
            The successfulness of the placement.
        """
        ndx = tuple(ndx)
        if self.query(agent, ndx):
            self._internal[ndx][agent.id] = agent
            agent.position = np.array(ndx)
            return True
        else:
            return False

    def remove(self, agent, ndx):
        """
        Remove an agent from an index.

        Args:
            agent: The agent to remove
            ndx: The old index for this agent
        """
        ndx = tuple(ndx)
        del self._internal[ndx][agent.id]

    def __getitem__(self, subscript):
        return self._internal.__getitem__(subscript)
