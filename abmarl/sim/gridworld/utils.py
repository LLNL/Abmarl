
import numpy as np

def create_mask(agent, mask_range, agents):
    """
    Generate a grid mask. 
    
    View-blocking agents can mask the grid from other agent's, restricting their
    ability to observe, attack, move, etc. We calculate the blocking by drawing
    rays from the center of the agent's position to the edges of the other agents'
    cell. All cells that are "behind" that cell and between the two rays are
    invisible to the observing agent. In the mask, 1 means that the cell is visibile,
    0 means that it is invisible.

    Args:
        agent: The agent of interest.
        mask_range: The integer range from the agent of interest.
        agents: The dictionary of agents.

    Returns:
        A grid of size (2 * range + 1) x (2 * range + 1) that shows which cells
        are masked to the agent of interest.
    """
    mask = np.ones((2 * mask_range + 1, 2 * mask_range + 1))
    for other in agents.values():
        if other.view_blocking:
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
