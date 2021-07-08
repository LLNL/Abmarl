# Maze generator -- Randomized Prim Algorithm

## Imports
import random



def generate_maze(rows, cols, wall_char='W', target_char='T', agents=None):
    if agents is None:
        agents = []

    cell_char = 0
    unvisited_char = 'u'
    maze = []

    # Find number of surrounding cells
    def surroundingCells(rand_wall):
        s_cells = 0
        if (maze[rand_wall[0]-1][rand_wall[1]] == cell_char):
            s_cells += 1
        if (maze[rand_wall[0]+1][rand_wall[1]] == cell_char):
            s_cells += 1
        if (maze[rand_wall[0]][rand_wall[1]-1] == cell_char):
            s_cells +=1
        if (maze[rand_wall[0]][rand_wall[1]+1] == cell_char):
            s_cells += 1
        return s_cells

    # Denote all cells as unvisited
    for i in range(0, rows):
        line = []
        for j in range(0, cols):
            line.append(unvisited_char)
        maze.append(line)

    # Set starting point as the middle
    starting_row = int(rows/2)
    starting_cell = int(cols/2)

    # Mark it as cell and add surrounding walls to the list
    maze[starting_row][starting_cell] = cell_char
    walls = []
    walls.append([starting_row - 1, starting_cell])
    walls.append([starting_row, starting_cell - 1])
    walls.append([starting_row, starting_cell + 1])
    walls.append([starting_row + 1, starting_cell])

    # Denote walls in maze
    maze[starting_row-1][starting_cell] = wall_char
    maze[starting_row][starting_cell - 1] = wall_char
    maze[starting_row][starting_cell + 1] = wall_char
    maze[starting_row + 1][starting_cell] = wall_char

    # Track the cells in the maze too
    cells = []

    while (walls):
        # Pick a random wall
        rand_wall = walls[int(random.random()*len(walls))-1]

        # Check if it is a left wall
        if (rand_wall[1] != 0):
            if (maze[rand_wall[0]][rand_wall[1]-1] == unvisited_char and maze[rand_wall[0]][rand_wall[1]+1] == cell_char):
                # Find the number of surrounding cells
                s_cells = surroundingCells(rand_wall)

                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = cell_char
                    cells.append([rand_wall[0], rand_wall[1]])

                    # Mark the new walls
                    # Upper cell
                    if (rand_wall[0] != 0):
                        if (maze[rand_wall[0]-1][rand_wall[1]] != cell_char):
                            maze[rand_wall[0]-1][rand_wall[1]] = wall_char
                        if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]-1, rand_wall[1]])


                    # Bottom cell
                    if (rand_wall[0] != rows-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != cell_char):
                            maze[rand_wall[0]+1][rand_wall[1]] = wall_char
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])

                    # Leftmost cell
                    if (rand_wall[1] != 0):	
                        if (maze[rand_wall[0]][rand_wall[1]-1] != cell_char):
                            maze[rand_wall[0]][rand_wall[1]-1] = wall_char
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])
                

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)

                continue

        # Check if it is an upper wall
        if (rand_wall[0] != 0):
            if (maze[rand_wall[0]-1][rand_wall[1]] == unvisited_char and maze[rand_wall[0]+1][rand_wall[1]] == cell_char):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = cell_char
                    cells.append([rand_wall[0], rand_wall[1]])

                    # Mark the new walls
                    # Upper cell
                    if (rand_wall[0] != 0):
                        if (maze[rand_wall[0]-1][rand_wall[1]] != cell_char):
                            maze[rand_wall[0]-1][rand_wall[1]] = wall_char
                        if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]-1, rand_wall[1]])

                    # Leftmost cell
                    if (rand_wall[1] != 0):
                        if (maze[rand_wall[0]][rand_wall[1]-1] != cell_char):
                            maze[rand_wall[0]][rand_wall[1]-1] = wall_char
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])

                    # Rightmost cell
                    if (rand_wall[1] != cols-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != cell_char):
                            maze[rand_wall[0]][rand_wall[1]+1] = wall_char
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)

                continue

        # Check the bottom wall
        if (rand_wall[0] != rows-1):
            if (maze[rand_wall[0]+1][rand_wall[1]] == unvisited_char and maze[rand_wall[0]-1][rand_wall[1]] == cell_char):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = cell_char
                    cells.append([rand_wall[0], rand_wall[1]])

                    # Mark the new walls
                    if (rand_wall[0] != rows-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != cell_char):
                            maze[rand_wall[0]+1][rand_wall[1]] = wall_char
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])
                    if (rand_wall[1] != 0):
                        if (maze[rand_wall[0]][rand_wall[1]-1] != cell_char):
                            maze[rand_wall[0]][rand_wall[1]-1] = wall_char
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])
                    if (rand_wall[1] != cols-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != cell_char):
                            maze[rand_wall[0]][rand_wall[1]+1] = wall_char
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)


                continue

        # Check the right wall
        if (rand_wall[1] != cols-1):
            if (maze[rand_wall[0]][rand_wall[1]+1] == unvisited_char and maze[rand_wall[0]][rand_wall[1]-1] == cell_char):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = cell_char
                    cells.append([rand_wall[0], rand_wall[1]])

                    # Mark the new walls
                    if (rand_wall[1] != cols-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != cell_char):
                            maze[rand_wall[0]][rand_wall[1]+1] = wall_char
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])
                    if (rand_wall[0] != rows-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != cell_char):
                            maze[rand_wall[0]+1][rand_wall[1]] = wall_char
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])
                    if (rand_wall[0] != 0):	
                        if (maze[rand_wall[0]-1][rand_wall[1]] != cell_char):
                            maze[rand_wall[0]-1][rand_wall[1]] = wall_char
                        if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]-1, rand_wall[1]])

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)

                continue

        # Delete the wall from the list anyway
        for wall in walls:
            if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                walls.remove(wall)

    # Set the starting cell as the target
    maze[starting_row][starting_cell] = target_char

    # Mark the remaining unvisited cells as walls
    for i in range(0, rows):
        for j in range(0, cols):
            if (maze[i][j] == unvisited_char):
                maze[i][j] = wall_char

    # Add agents to the maze
    for agent in agents:
        cell = random.choice(cells)
        maze[cell[0]][cell[1]] = agent
        cells.remove(cell)


    # Print final maze
    for i in range(0, rows):
        for j in range(0, cols):
            print(maze[i][j], end=" ")
        print('\n')

generate_maze(7, 7, agents=['A']*4)
