
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
        if other.blocking:
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
