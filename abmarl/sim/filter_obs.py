
import numpy as np

view = 8
grid = np.ones((2 * view + 1, 2 * view + 1))
grid[view, view] = 0 # agent's location

r_diffs = [3, 0, -2, -3, 0, -3, -4]
c_diffs = [4, -3, 3, -2, -5, 3, 3]

for i in range(len(r_diffs)):
    r_diff = r_diffs[i]
    c_diff = c_diffs[i]

    if c_diff > 0 and r_diff == 0: # Wall is to the right of agent
        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
        for c in range(c_diff, view+1):
            for r in range(-view, view+1):
                if c == c_diff and r == r_diff: continue # don't mask the wall
                if lower(c) < r < upper(c):
                    grid[r + view, c + view] = 0


    elif c_diff > 0 and r_diff > 0: # Wall is below-right of agent
        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
        for c in range(c_diff, view+1):
            for r in range(r_diff, view+1):
                if c == c_diff and r == r_diff: continue # Don't mask the wall
                if lower(c) < r < upper(c):
                    grid[r + view, c + view] = 0


    elif c_diff == 0 and r_diff > 0: # Wall is below the agent
        left = lambda t: (c_diff - 0.5) / (r_diff - 0.5) * t
        right = lambda t: (c_diff + 0.5) / (r_diff - 0.5) * t
        for c in range(-view, view+1):
            for r in range(r_diff, view+1):
                if c == c_diff and r == r_diff: continue # don't mask the wall
                if left(r) < c < right(r):
                    grid[r + view, c + view] = 0


    elif c_diff < 0 and r_diff > 0: # Wall is below-left of agent
        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
        for c in range(c_diff, -view-1, -1):
            for r in range(r_diff, view+1):
                if c == c_diff and r == r_diff: continue # don't mask the wall
                if lower(c) < r < upper(c):
                    grid[r + view, c + view] = 0


    elif c_diff < 0 and r_diff == 0: # Wall is left of agent
        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
        for c in range(c_diff, -view-1, -1):
            for r in range(-view, view+1):
                if c == c_diff and r == r_diff: continue # don't mask the wall
                if lower(c) < r < upper(c):
                    grid[r + view, c + view] = 0


    elif c_diff < 0 and r_diff < 0: # Wall is above-left of agent
        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
        for c in range(c_diff, -view - 1, -1):
            for r in range(r_diff, -view - 1, -1):
                if c == c_diff and r == r_diff: continue # don't mask the wall
                if lower(c) < r < upper(c):
                    grid[r + view, c + view] = 0


    elif c_diff == 0 and r_diff < 0: # Wall is above the agent
        left = lambda t: (c_diff - 0.5) / (r_diff + 0.5) * t
        right = lambda t: (c_diff + 0.5) / (r_diff + 0.5) * t
        for c in range(-view, view+1):
            for r in range(r_diff, -view - 1, -1):
                if c == c_diff and r == r_diff: continue # don't mask the wall
                if left(r) < c < right(r):
                    grid[r + view, c + view] = 0


    elif c_diff > 0 and r_diff < 0: # Wall is above-right of agent
        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
        for c in range(c_diff, view+1):
            for r in range(r_diff, -view - 1, -1):
                if c == c_diff and r == r_diff: continue # don't mask the wall
                if lower(c) < r < upper(c):
                    grid[r + view, c + view] = 0

print(grid)

