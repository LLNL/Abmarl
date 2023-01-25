
import numpy as np
import pytest

from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.agent import GridWorldAgent


def test_grid_rows():
    grid = Grid(2, 6)
    assert grid.rows == 2

    with pytest.raises(AssertionError):
        Grid(-1, 3)


def test_grid_cols():
    grid = Grid(2, 6)
    assert grid.cols == 6

    with pytest.raises(AssertionError):
        Grid(1, 5.0)


def test_grid_internal():
    grid = Grid(2, 6)
    np.testing.assert_equal(grid._internal, np.empty((2, 6), dtype=object))


def test_grid_overlapping():
    grid = Grid(3, 3, overlapping={1: [2], 2: [1], 3: [1, 2]})
    assert grid._overlapping == {1: [2], 2: [1], 3: [1, 2]}

    with pytest.raises(AssertionError):
        Grid(2, 2, overlapping=[1, 2, 3])

    with pytest.raises(AssertionError):
        Grid(2, 2, overlapping={'1': [3], 2.0: [6]})

    with pytest.raises(AssertionError):
        Grid(2, 2, overlapping={1: 3, 2: [6]})

    with pytest.raises(AssertionError):
        Grid(2, 2, overlapping={1: ['2', 3], 2: [2, 3]})


def test_grid_reset():
    grid = Grid(3, 3)
    grid.reset()
    for i in range(grid.rows):
        for j in range(grid.cols):
            assert grid[i, j] == {}

    agent = GridWorldAgent(id='agent0', encoding=1)
    assert grid.place(agent, (1, 0))
    assert grid[1, 0] == {'agent0': agent}
    for i in range(grid.rows):
        for j in range(grid.cols):
            if i == 1 and j == 0:
                continue
            assert grid[i, j] == {}
    grid.reset()
    assert grid[1, 0] == {}


def test_grid_query():
    grid = Grid(3, 3)
    grid.reset()
    agent1 = GridWorldAgent(id='agent1', encoding=1)
    agent2 = GridWorldAgent(id='agent2', encoding=1)
    assert grid.place(agent1, (1, 0))
    assert not grid.query(agent2, (1, 0))
    assert grid.query(agent2, (0, 1))


def test_grid_query_with_overlap():
    grid = Grid(3, 3, overlapping={1: [1]})
    grid.reset()
    agent1 = GridWorldAgent(id='agent1', encoding=1)
    agent2 = GridWorldAgent(id='agent2', encoding=1)
    assert grid.place(agent1, (1, 0))
    assert grid.query(agent2, (1, 0))
    assert grid.query(agent1, (1, 0))


def test_grid_query_overlap_with_inactive_agent():
    grid = Grid(3, 3)
    grid.reset()
    agent1 = GridWorldAgent(id='agent1', encoding=1)
    agent2 = GridWorldAgent(id='agent2', encoding=1)
    assert grid.place(agent1, (1, 0))
    agent1._active = False
    assert not grid.query(agent2, (1, 0))
    grid.remove(agent1, agent1.position)
    assert grid.query(agent2, (1, 0))


def test_grid_place():
    grid = Grid(3, 3)
    grid.reset()
    agent1 = GridWorldAgent(id='agent1', encoding=1)
    agent2 = GridWorldAgent(id='agent2', encoding=2)
    assert grid.place(agent1, (1, 0))
    assert not grid.place(agent2, (1, 0))


def test_grid_place_with_overlap():
    grid = Grid(3, 3, overlapping={1: [2], 2: [1]})
    grid.reset()
    agent1 = GridWorldAgent(id='agent1', encoding=1)
    agent2 = GridWorldAgent(id='agent2', encoding=2)
    assert grid.place(agent1, (1, 0))
    assert grid.place(agent2, (1, 0))
    assert grid[1, 0] == {'agent1': agent1, 'agent2': agent2}


def test_grid_remove():
    grid = Grid(3, 3, overlapping={1: [2], 2: [1]})
    grid.reset()
    agent1 = GridWorldAgent(id='agent1', encoding=1)
    agent2 = GridWorldAgent(id='agent2', encoding=2)
    agent3 = GridWorldAgent(id='agent3', encoding=2)
    assert grid.place(agent1, (1, 0))
    assert grid.place(agent2, (1, 0))
    assert grid.place(agent3, (0, 1))
    assert grid[1, 0] == {'agent1': agent1, 'agent2': agent2}
    assert grid[0, 1] == {'agent3': agent3}

    grid.remove(agent1, (1, 0))
    grid.remove(agent3, (0, 1))
    assert grid[1, 0] == {'agent2': agent2}
    assert grid[0, 1] == {}

    with pytest.raises(KeyError):
        grid.remove(agent1, (1, 0))
