
import numpy as np

from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.state import PositionState, HealthState, StateBaseComponent
from abmarl.sim.gridworld.agent import HealthAgent, GridWorldAgent


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
