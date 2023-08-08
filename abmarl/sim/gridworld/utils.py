
import numpy as np


def create_grid_and_mask(agent, grid, mask_range, agents):
    """
    Generate a local grid and a mask.

    Create a local grid centered around the agents location and fill it will the
    values from the grid.

    Blocking agents can mask the grid from other agent's, restricting their
    ability to observe, attack, move, etc. We calculate the masking by drawing
    rays from the center of the agent's position to the edges of the other agents'
    cell. All cells that are "behind" that cell and between the two rays are
    invisible to the observing agent. In the mask, 1 means that the cell is visibile,
    0 means that it is invisible.

    Args:
        agent: The agent of interest.
        grid: The grid.
        mask_range: The integer range from the agent of interest.
        agents: The dictionary of agents.

    Returns:
        Two matrices. The first is a local grid centered around the agent's location
        with values from the actual grid. The second is a mask of size
        (2 * range + 1) x (2 * range + 1) that shows which cells are masked to the
        agent of interest.
    """
    # Generate a completely empty grid
    local_grid = np.empty((mask_range * 2 + 1, mask_range * 2 + 1), dtype=object)

    # Copy the section of the grid around the agent's position
    (r, c) = agent.position
    r_lower = max([0, r - mask_range])
    r_upper = min([grid.rows - 1, r + mask_range]) + 1
    c_lower = max([0, c - mask_range])
    c_upper = min([grid.cols - 1, c + mask_range]) + 1
    local_grid[
        (r_lower+mask_range-r):(r_upper+mask_range-r),
        (c_lower+mask_range-c):(c_upper+mask_range-c)
    ] = grid[r_lower:r_upper, c_lower:c_upper]

    mask = np.ones((2 * mask_range + 1, 2 * mask_range + 1))
    for other in agents.values():
        if other.active and other.blocking:
            r_diff, c_diff = other.position - agent.position
            # Ensure the other agent within the view range
            if -mask_range <= r_diff <= mask_range and \
                    -mask_range <= c_diff <= mask_range:
                if c_diff > 0 and r_diff == 0: # Other is to the right of agent
                    upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                    lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                    for c in range(c_diff, mask_range+1):
                        for r in range(-mask_range, mask_range+1):
                            if c == c_diff and r == r_diff: continue # don't mask the other
                            if lower(c) < r < upper(c):
                                mask[r + mask_range, c + mask_range] = 0
                elif c_diff > 0 and r_diff > 0: # Other is below-right of agent
                    upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                    lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                    for c in range(c_diff, mask_range+1):
                        for r in range(r_diff, mask_range+1):
                            if c == c_diff and r == r_diff: continue # Don't mask the other
                            if lower(c) < r < upper(c):
                                mask[r + mask_range, c + mask_range] = 0
                elif c_diff == 0 and r_diff > 0: # Other is below the agent
                    left = lambda t: (c_diff - 0.5) / (r_diff - 0.5) * t
                    right = lambda t: (c_diff + 0.5) / (r_diff - 0.5) * t
                    for c in range(-mask_range, mask_range+1):
                        for r in range(r_diff, mask_range+1):
                            if c == c_diff and r == r_diff: continue # don't mask the other
                            if left(r) < c < right(r):
                                mask[r + mask_range, c + mask_range] = 0
                elif c_diff < 0 and r_diff > 0: # Other is below-left of agent
                    upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                    lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                    for c in range(c_diff, -mask_range-1, -1):
                        for r in range(r_diff, mask_range+1):
                            if c == c_diff and r == r_diff: continue # don't mask the other
                            if lower(c) < r < upper(c):
                                mask[r + mask_range, c + mask_range] = 0
                elif c_diff < 0 and r_diff == 0: # Other is left of agent
                    upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                    lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                    for c in range(c_diff, -mask_range-1, -1):
                        for r in range(-mask_range, mask_range+1):
                            if c == c_diff and r == r_diff: continue # don't mask the other
                            if lower(c) < r < upper(c):
                                mask[r + mask_range, c + mask_range] = 0
                elif c_diff < 0 and r_diff < 0: # Other is above-left of agent
                    upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
                    lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
                    for c in range(c_diff, -mask_range - 1, -1):
                        for r in range(r_diff, -mask_range - 1, -1):
                            if c == c_diff and r == r_diff: continue # don't mask the other
                            if lower(c) < r < upper(c):
                                mask[r + mask_range, c + mask_range] = 0
                elif c_diff == 0 and r_diff < 0: # Other is above the agent
                    left = lambda t: (c_diff - 0.5) / (r_diff + 0.5) * t
                    right = lambda t: (c_diff + 0.5) / (r_diff + 0.5) * t
                    for c in range(-mask_range, mask_range+1):
                        for r in range(r_diff, -mask_range - 1, -1):
                            if c == c_diff and r == r_diff: continue # don't mask the other
                            if left(r) < c < right(r):
                                mask[r + mask_range, c + mask_range] = 0
                elif c_diff > 0 and r_diff < 0: # Other is above-right of agent
                    upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
                    lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
                    for c in range(c_diff, mask_range+1):
                        for r in range(r_diff, -mask_range - 1, -1):
                            if c == c_diff and r == r_diff: continue # don't mask the other
                            if lower(c) < r < upper(c):
                                mask[r + mask_range, c + mask_range] = 0

    return local_grid, mask


def generate_maze(rows, cols, start=None):
    """
    Generate a maze using Prim's Algorithm.

    Args:
        rows: The number of rows in the maze.
        cols: The number of columns in the maze.
        start: Numpy array, starting cell of the maze. If not specified, then a
            random cell will be chosen.

    Returns:
        A maze represented by a numpy array, where 0 is a passage and 1 a wall.
    """
    def unvisited_neighboring_cells(cell):
        """
        Determine which of the neighboring cells is unvisted and marks them as walls.

        Args:
            cell: The cell in question.

        Returns:
            List of unvisted neighboring cells now marked as walls.
        """
        neighboring_cells = []
        for neighbor in [
            (cell[0] - 1, cell[1]),
            (cell[0] + 1, cell[1]),
            (cell[0], cell[1] - 1),
            (cell[0], cell[1] + 1),
        ]:
            if neighbor[0] in [0, rows - 1] or neighbor[1] in [0, cols - 1]:
                continue # Don't include neighbors along the borders
            if grid[tuple(neighbor)] == 2: # Unvisited
                neighboring_cells.append(neighbor)
                grid[tuple(neighbor)] = 1
        return neighboring_cells

    def sum_neighboring_free(cell):
        """
        Calculates the number of neighboring cells which are passages.

        We add this as a check to avoid open spaces in the maze.

        Args:
            cell: The cell in question.

        Returns:
            The nubmer of neighboring cells that are passages.
        """
        return sum([
            grid[neighbor] == 0
            for neighbor in [
                (cell[0] - 1, cell[1]),
                (cell[0] + 1, cell[1]),
                (cell[0], cell[1] - 1),
                (cell[0], cell[1] + 1),
            ]
        ])

    assert type(rows) is int and rows > 0, "Rows must be a positive integer."
    assert type(cols) is int and cols > 0, "Columns must be a positive integer."
    # We tack on two rows and columns because we will remove the borders
    rows += 2
    cols += 2
    grid = 2 * np.ones((rows, cols))

    if start is not None:
        assert type(start) is np.ndarray, "Starting cell must be a numpy array."
        assert start.shape == (2,), "Starting cell must be a 2D coordinate."
        start = start + 1 # Need to increment the starting cell because we added borders
        # NOTE: We must not do start += 1 because that will affect the array outside
        # the scope of this function.
    else:
        start = np.random.randint(1, np.array(grid.shape) - 1) # Don't chose a border cell
    grid[tuple(start)] = 0

    unvisited_walls = unvisited_neighboring_cells(start)
    while unvisited_walls:
        current_cell = unvisited_walls[np.random.randint(0, len(unvisited_walls))]
        if ((grid[current_cell[0] - 1, current_cell[1]] == 2) ^
            (grid[current_cell[0] + 1, current_cell[1]] == 2)) or \
           ((grid[current_cell[0], current_cell[1] - 1] == 2) ^
            (grid[current_cell[0], current_cell[1] + 1] == 2)):

            if sum_neighboring_free(current_cell) < 2:
                grid[tuple(current_cell)] = 0
                unvisited_walls = list(set(
                    unvisited_walls + unvisited_neighboring_cells(current_cell)
                ))
        unvisited_walls.remove(current_cell)

    grid[grid == 2] = 1 # Convert unvisited cells to walls
    return grid[1:-1, 1:-1] # Lop of the borders
