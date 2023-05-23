
import numpy as np
import pytest

from abmarl.sim.gridworld.utils import generate_maze


def test_generate_maze():
    np.random.seed(24)
    maze = generate_maze(5, 9, start=np.array([1, 1]))

    assert type(maze) is np.ndarray
    assert maze.shape[0] == 5
    assert maze.shape[1] == 9
    assert sum([el for el in np.nditer(maze) if el not in [0, 1]]) == 0
    assert maze[1, 1] == 0

    with pytest.raises(AssertionError):
        generate_maze(0, 4)

    with pytest.raises(AssertionError):
        generate_maze(3, -1)

    with pytest.raises(AssertionError):
        generate_maze(3, 3, [0, 1])

    with pytest.raises(AssertionError):
        generate_maze(3, 3, np.array([0]))

    with pytest.raises(IndexError):
        generate_maze(3, 3, np.array([6, 13]))


def test_generate_maze_no_start():
    np.random.seed(24)
    maze = generate_maze(4, 4)

    assert type(maze) is np.ndarray
    assert maze.shape[0] == 4
    assert maze.shape[1] == 4
    assert sum([el for el in np.nditer(maze) if el not in [0, 1]]) == 0
