
import numpy as np


def build_grid_sim(file_name, object_registry):
    """
    Build a custom grid with agents.

    Args:
        file_name: Name of the file that specifies the grid. In the file, each
            cell should be a single alphanumeric character indicating which agent
            will be there. That agent will be given that initial position and start
            there at the beginning of each episode. 0's are reserved for empty
            space. For example:
            0 A W W W A 0
            0 0 0 0 0 0 0
            0 A W W W A 0
            will create a 3-by-7 grid with some agents along the top and bottom
            of the grid and another type of agent in the corner.
        object_registry: A dictionary that maps characters from the file to a
            function that generates the agent. This must be a function because
            each agent must have unique id, which is generated here. For example,
            using the grid above and some pre-created Agent classes:
            {
                'A': lambda n: ExploringAgent(id=f'explorer{n}', view_range=3, move_range=1),
                'W': lambda n: WallAgent(id=f'wall{n}')
            }
            will create a grid with ExploringAgents in the corners and WallAgents
            along the top and bottom rows.

    Returns:
        A 3-element dict. The first element is the number of rows in the grid.
        The second element is the number of columns in the grid. The third element
        is the dictionary of agents.
    """
    assert 0 not in object_registry, "0 is reserved for empty space."
    agents = {}
    n = 0
    with open(file_name, 'r') as fp:
        lines = fp.read().splitlines()
        cols = len(lines[0].split(' '))
        rows = len(lines)
        for row, line in enumerate(lines):
            chars = line.split(' ')
            assert len(chars) == cols, f"Mismatched number of columns per row in {file_name}"
            for col, char in enumerate(chars):
                if char in object_registry:
                    agent = object_registry[char](n)
                    agent.initial_position = np.array([row, col])
                    agents[agent.id] = agent
                    n += 1

    return {'rows': rows, 'cols': cols, 'agents': agents}
