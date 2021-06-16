
import numpy as np

from abmarl.sim.grid_world import GridWorldBaseComponent

class GridWorldState(GridWorldBaseComponent):
    """
    Manage the agent's positions and the grid.
    """
    def __init__(self, rows=None, cols=None, **kwargs):
        self.rows = rows
        self.cols = cols

    def reset(self, **kwargs):
        """
        Give agents their starting positions.

        We use the agent's initial position if it exists. Otherwise, we randomly
        place the agent in the grid. Every agent will occupy a unique square.
        """
        # Grid lookup by position
        self.grid = np.empty((self.rows, self.cols), dtype=object)
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
                    np.ravel_multi_index(agent.initial_position, (self.rows, self.cols))
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
