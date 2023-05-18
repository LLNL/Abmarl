
import numpy as np

from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.state import PositionState, MazePlacementState, HealthState, \
     StateBaseComponent
from abmarl.sim.gridworld.agent import HealthAgent, GridWorldAgent
import pytest


def test_position_state():
    grid = Grid(3, 3)
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 1])),
        'agent1': GridWorldAgent(id='agent1', encoding=1, initial_position=np.array([1, 2])),
        'agent2': GridWorldAgent(id='agent2', encoding=1, initial_position=np.array([2, 0]))
    }

    position_state = PositionState(grid=grid, agents=agents)
    assert isinstance(position_state, StateBaseComponent)
    position_state.reset()

    np.testing.assert_equal(agents['agent0'].position, np.array([0, 1]))
    np.testing.assert_equal(agents['agent1'].position, np.array([1, 2]))
    np.testing.assert_equal(agents['agent2'].position, np.array([2, 0]))
    assert grid[0, 1] == {'agent0': agents['agent0']}
    assert grid[1, 2] == {'agent1': agents['agent1']}
    assert grid[2, 0] == {'agent2': agents['agent2']}


def test_position_state_small_grid():
    grid = Grid(1, 2, overlapping={1: {1, 2}, 2: {1, 2}, 3: {3}})
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 0])),
        'agent1': GridWorldAgent(id='agent1', encoding=2, initial_position=np.array([0, 0])),
        'agent2': GridWorldAgent(id='agent2', encoding=3),
        'agent3': GridWorldAgent(id='agent3', encoding=3),
        'agent4': GridWorldAgent(id='agent4', encoding=2),
        'agent5': GridWorldAgent(id='agent5', encoding=1)
    }
    # Encoding 3 can only go on (0, 1) because (0, 0) is taken and can't be overlapped.
    # If agents 2 and 3 were placed after 4 and 5, they would likely not have a
    # cell, so we see that the order of the agents matters in their initial placement.
    position_state = PositionState(grid=grid, agents=agents)
    position_state.reset()
    assert 'agent0' in grid[0, 0]
    assert 'agent1' in grid[0, 0]
    assert 'agent2' in grid[0, 1]
    assert 'agent3' in grid[0, 1]
    assert 'agent4' in grid[0, 0]
    assert 'agent5' in grid[0, 0]


    # This will fail because agents 0 and 1 have taken all available cells, so there
    # is no where to put encoding 3
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 0])),
        'agent1': GridWorldAgent(id='agent1', encoding=2, initial_position=np.array([0, 1])),
        'agent2': GridWorldAgent(id='agent2', encoding=3)
    }
    position_state = PositionState(grid=grid, agents=agents)
    with pytest.raises(RuntimeError):
        position_state.reset()


    # This may fail because agent 1 might have taken the last avilable cell.
    # We have set two seeds: one where it passes and another where it fails
    np.random.seed(24)
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 0])),
        'agent1': GridWorldAgent(id='agent1', encoding=2),
        'agent2': GridWorldAgent(id='agent2', encoding=3),
    }
    position_state = PositionState(grid=grid, agents=agents)
    position_state.reset()
    assert 'agent0' in grid[0, 0]
    assert 'agent1' in grid[0, 0]
    assert 'agent2' in grid[0, 1]

    np.random.seed(17)
    with pytest.raises(RuntimeError):
        position_state.reset()


def test_health_state():
    grid = Grid(3, 3)
    agents = {
        'agent0': HealthAgent(id='agent0', encoding=1, initial_health=0.24),
        'agent1': HealthAgent(id='agent1', encoding=1),
        'agent2': HealthAgent(id='agent2', encoding=1)
    }

    health_state = HealthState(agents=agents, grid=grid)
    assert isinstance(health_state, StateBaseComponent)
    health_state.reset()

    assert agents['agent0'].health == 0.24
    assert 0 <= agents['agent1'].health <= 1
    assert 0 <= agents['agent2'].health <= 1
    assert agents['agent0'].active
    assert agents['agent1'].active
    assert agents['agent2'].active


def test_maze_placement_state():
    target_agent = GridWorldAgent(id='target', encoding=1, render_color='g')
    ip_agent = GridWorldAgent(
        id='ip_agent',
        encoding=1,
        initial_position=np.array([3, 2]),
        render_color='b'
    )
    barrier_agents = {
        f'barrier_agent{i}': GridWorldAgent(
            id=f'barrier_agent{i}',
            encoding=2
        ) for i in range(5)
    }
    free_agents = {
        f'free_agent{i}': GridWorldAgent(
            id=f'free_agent{i}',
            encoding=3
        ) for i in range(3)
    }
    agents={
        'target': target_agent,
        'ip_agent': ip_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(4, 4, overlapping={1: {3}, 3: {3}})
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3}
    )

    state.reset()
test_maze_placement_state()