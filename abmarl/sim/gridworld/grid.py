
import numpy as np

class Grid:
    """
    A Grid has a numpy array that stores the agents at indicies of the array.

    Components can interface with the internals of the Grid. Each index in the
    grid is a dictionary that maps the agent id to the agent object itself. If agents
    can overlap, then there may be more than one agent per cell.

    Args:
        rows: The number of rows in the grid.
        cols: The number of columns in the grid.
    """
    def __init__(self, rows, cols):
        self._internal = np.empty((self.rows, self.cols), dtype=object)

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

    def reset(self, **kwargs):
        """
        Reset the grid to an empty state.
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self._internal[i,j] = {}

    def query(self, agent, ndx):
        """
        Query a cell in the grid to see if is available to this agent. The cell
        is available for the agent if it is empty or if both the occupying agent
        and the querying agent are overlappable.

        Args:
            agent: The agent for which we are checking availabilty.
            ndx: The cell to query.

        Returns:
            True if the cell is available for this agent, otherwise False.
        """
        ndx = tuple(ndx)
        return not self._internal[ndx] or (
            next(iter(self._internal[ndx].values())).overlappable and agent.overlappable
        )

    def place(self, agent, ndx):
        """
        Place an agent at an index.

        If the placement is successful, the agent will be placed at that index
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
