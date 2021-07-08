# Maze generator -- Randomized Prim Algorithm

## Imports
import random



def generate_maze(rows, cols, wall='w', target='T', agents=None):
    ## Main code
    # Init variables
    cell = 'c'
    unvisited = 'u'
    maze = []

    # Find number of surrounding cells
    def surroundingCells(rand_wall):
        s_cells = 0
        if (maze[rand_wall[0]-1][rand_wall[1]] == 'c'):
            s_cells += 1
        if (maze[rand_wall[0]+1][rand_wall[1]] == 'c'):
            s_cells += 1
        if (maze[rand_wall[0]][rand_wall[1]-1] == 'c'):
            s_cells +=1
        if (maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
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
    maze[starting_height][starting_width] = cell
    walls = []
    walls.append([starting_height - 1, starting_width])
    walls.append([starting_height, starting_width - 1])
    walls.append([starting_height, starting_width + 1])
    walls.append([starting_height + 1, starting_width])

    # Denote walls in maze
    maze[starting_height-1][starting_width] = 'w'
    maze[starting_height][starting_width - 1] = 'w'
    maze[starting_height][starting_width + 1] = 'w'
    maze[starting_height + 1][starting_width] = 'w'

    while (walls):
        # Pick a random wall
        rand_wall = walls[int(random.random()*len(walls))-1]

        # Check if it is a left wall
        if (rand_wall[1] != 0):
            if (maze[rand_wall[0]][rand_wall[1]-1] == 'u' and maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
                # Find the number of surrounding cells
                s_cells = surroundingCells(rand_wall)

                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = 'c'

                    # Mark the new walls
                    # Upper cell
                    if (rand_wall[0] != 0):
                        if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]-1, rand_wall[1]])


                    # Bottom cell
                    if (rand_wall[0] != rows-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])

                    # Leftmost cell
                    if (rand_wall[1] != 0):	
                        if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])
                

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)

                continue

        # Check if it is an upper wall
        if (rand_wall[0] != 0):
            if (maze[rand_wall[0]-1][rand_wall[1]] == 'u' and maze[rand_wall[0]+1][rand_wall[1]] == 'c'):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = 'c'

                    # Mark the new walls
                    # Upper cell
                    if (rand_wall[0] != 0):
                        if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]-1, rand_wall[1]])

                    # Leftmost cell
                    if (rand_wall[1] != 0):
                        if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])

                    # Rightmost cell
                    if (rand_wall[1] != cols-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)

                continue

        # Check the bottom wall
        if (rand_wall[0] != rows-1):
            if (maze[rand_wall[0]+1][rand_wall[1]] == 'u' and maze[rand_wall[0]-1][rand_wall[1]] == 'c'):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = 'c'

                    # Mark the new walls
                    if (rand_wall[0] != rows-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])
                    if (rand_wall[1] != 0):
                        if (maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                        if ([rand_wall[0], rand_wall[1]-1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]-1])
                    if (rand_wall[1] != cols-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])

                # Delete wall
                for wall in walls:
                    if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                        walls.remove(wall)


                continue

        # Check the right wall
        if (rand_wall[1] != cols-1):
            if (maze[rand_wall[0]][rand_wall[1]+1] == 'u' and maze[rand_wall[0]][rand_wall[1]-1] == 'c'):

                s_cells = surroundingCells(rand_wall)
                if (s_cells < 2):
                    # Denote the new path
                    maze[rand_wall[0]][rand_wall[1]] = 'c'

                    # Mark the new walls
                    if (rand_wall[1] != cols-1):
                        if (maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                            maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                        if ([rand_wall[0], rand_wall[1]+1] not in walls):
                            walls.append([rand_wall[0], rand_wall[1]+1])
                    if (rand_wall[0] != rows-1):
                        if (maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                        if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                            walls.append([rand_wall[0]+1, rand_wall[1]])
                    if (rand_wall[0] != 0):	
                        if (maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                            maze[rand_wall[0]-1][rand_wall[1]] = 'w'
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
                maze[i][j] = 'w'

    # Set entrance and exit
    for i in range(0, cols):
        if (maze[1][i] == 'c'):
            maze[0][i] = 'c'
            break

    for i in range(cols-1, 0, -1):
        if (maze[rows-2][i] == 'c'):
            maze[rows-1][i] = 'c'
            break

    # Print final maze
    for i in range(0, rows):
        for j in range(0, cols):
            print(maze[i][j], end=" ")
        print('\n')

generate_maze(5, 5)
