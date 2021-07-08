# Maze generator -- Randomized Prim Algorithm

## Imports
import random



def generate_maze(rows, cols, wall_char='W', target='T', agents=None):
    ## Main code
    # Init variables
    cell_char = 0
    unvisited = 'u'
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
            line.append(unvisited)
        maze.append(line)

    # Randomize starting point and set it a cell
    starting_height = int(random.random()*rows)
    starting_width = int(random.random()*cols)
    if (starting_height == 0):
        starting_height += 1
    if (starting_height == rows-1):
        starting_height -= 1
    if (starting_width == 0):
        starting_width += 1
    if (starting_width == cols-1):
        starting_width -= 1

    # Mark it as cell and add surrounding walls to the list
    maze[starting_height][starting_width] = cell_char
    walls = []
    walls.append([starting_height - 1, starting_width])
    walls.append([starting_height, starting_width - 1])
    walls.append([starting_height, starting_width + 1])
    walls.append([starting_height + 1, starting_width])

    # Denote walls in maze
    maze[starting_height-1][starting_width] = wall_char
    maze[starting_height][starting_width - 1] = wall_char
    maze[starting_height][starting_width + 1] = wall_char
    maze[starting_height + 1][starting_width] = wall_char

    while (walls):
        # Pick a random wall
        rand_wall = walls[int(random.random()*len(walls))-1]

        # Check if it is a left wall
        if (rand_wall[1] != 0):
            if (maze[rand_wall[0]][rand_wall[1]-1] == 'u' and maze[rand_wall[0]][rand_wall[1]+1] == cell_char):
                # Find the number of surrounding cells
                s_cells = surroundingCells(rand_wall)

                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = cell_char

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
            if (maze[rand_wall[0]-1][rand_wall[1]] == 'u' and maze[rand_wall[0]+1][rand_wall[1]] == cell_char):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = cell_char

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
            if (maze[rand_wall[0]+1][rand_wall[1]] == 'u' and maze[rand_wall[0]-1][rand_wall[1]] == cell_char):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = cell_char

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
            if (maze[rand_wall[0]][rand_wall[1]+1] == 'u' and maze[rand_wall[0]][rand_wall[1]-1] == cell_char):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = cell_char

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
        


    # Mark the remaining unvisited cells as walls
    for i in range(0, rows):
        for j in range(0, cols):
            if (maze[i][j] == 'u'):
                maze[i][j] = wall_char

    # Print final maze
    for i in range(0, rows):
        for j in range(0, cols):
            print(maze[i][j], end=" ")
        print('\n')

generate_maze(5, 5)
