
import numpy as np

view = 8
grid = np.ones((2 * view + 1, 2 * view + 1))
grid[view, view] = 0 # agent's location

x_diffs = [4, -3, 3, -2, -5]
y_diffs = [-3, 0, 2, 3, 0]

for i in range(1):
    # x_diff = x_diffs[i]
    # y_diff = y_diffs[i]

    r_diff = 2
    c_diff = 4

    if grid[r_diff + view, c_diff + view] != 1: continue # I cannot see the wall because its behind another wall

    if c_diff > 0 and r_diff == 0: # Wall to the right of agent
        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
        for c in range(c_diff, view+1):
            for r in range(-view, view+1):
                if lower(c) < r < upper(c):
                    grid[r + view, c + view] = 0


    elif c_diff > 0 and r_diff > 0: # Wall below-right of agent
        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
        for c in range(c_diff, view+1):
            for r in range(r_diff, view+1):
                if lower(c) < r < upper(c):
                    grid[r + view, c + view] = 0


    elif c_diff == 0 and r_diff > 0:
        left = lambda t: (c_diff - 0.5) / (r_diff - 0.5) * t
        right = lambda t: (c_diff + 0.5) / (r_diff - 0.5) * t
        for x in range(-view, view+1):
            for y in range(r_diff, view+1):
                if left(y) < x < right(y):
                    grid[x + view, y + view] = 0


    elif c_diff < 0 and r_diff > 0:
        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
        for x in range(c_diff, -view-1, -1):
            for y in range(r_diff, view+1):
                if lower(x) < y < upper(x):
                    grid[x + view, y + view] = 0


    elif c_diff < 0 and r_diff == 0:
        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
        for x in range(c_diff, -view-1, -1):
            for y in range(-view, view+1):
                if lower(x) < y < upper(x):
                    grid[x + view, y + view] = 0


    elif c_diff < 0 and r_diff < 0:
        upper = lambda t: (r_diff + 0.5) / (c_diff - 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff + 0.5) * t
        for x in range(c_diff, -view - 1, -1):
            for y in range(r_diff, -view - 1, -1):
                if lower(x) < y < upper(x):
                    grid[x + view, y + view] = 0


    elif c_diff == 0 and r_diff < 0:
        left = lambda t: (c_diff - 0.5) / (r_diff + 0.5) * t
        right = lambda t: (c_diff + 0.5) / (r_diff + 0.5) * t
        for x in range(-view, view+1):
            for y in range(r_diff, -view - 1, -1):
                if left(y) < x < right(y):
                    grid[x + view, y + view] = 0


    elif c_diff > 0 and r_diff < 0:
        upper = lambda t: (r_diff + 0.5) / (c_diff + 0.5) * t
        lower = lambda t: (r_diff - 0.5) / (c_diff - 0.5) * t
        for x in range(c_diff, view+1):
            for y in range(r_diff, -view - 1, -1):
                if lower(x) < y < upper(x):
                    grid[x + view, y + view] = 0
    
    grid[r_diff + view, c_diff + view] = 2

# print(np.flipud(np.transpose(grid)))
print(grid)

