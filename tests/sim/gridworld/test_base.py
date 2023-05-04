
import numpy as np
import pytest

from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.examples.sim import MultiAgentGridSim
    
def test_build():
    sim = MultiAgentGridSim.build_sim(3, 4, agents={})
    assert sim.agents == {}
    assert isinstance(sim.grid, Grid)
    assert sim.grid.rows == 3
    assert sim.grid.cols == 4
    np.testing.assert_array_equal(sim.grid._internal, np.empty((3, 4), dtype=object))

    sim.reset()
    np.testing.assert_array_equal(
        sim.grid._internal,  np.array([
            [{}, {}, {}, {}],
            [{}, {}, {}, {}],
            [{}, {}, {}, {}]
        ])
    )

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim(3.0, 4)

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim(0, 4)

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim(3, -4)

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim(3, '4')

def test_build_from_grid():
    grid = Grid(2, 2)
    grid.reset()
    agents = [
        GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 0])),
        GridWorldAgent(id='agent1', encoding=1, initial_position=np.array([0, 1])),
        GridWorldAgent(id='agent2', encoding=1, initial_position=np.array([1, 0])),
        GridWorldAgent(id='agent3', encoding=1, initial_position=np.array([1, 1])),
    ]
    # Place 0th and 3rd agent in the grid according to their initial positions.
    # Place 1st and 2nd agent in the grid swapped. We want to test how the
    # initial positions gets defined during the build process.
    grid.place(agents[0], (0, 0))
    grid.place(agents[1], (0, 1))
    grid.place(agents[2], (1, 0))
    grid.place(agents[3], (1, 1))

    sim = MultiAgentGridSim.build_sim_from_grid(grid)
    assert sim.agents == {
        'agent0': agents[0],
        'agent1': agents[1],
        'agent2': agents[2],
        'agent3': agents[3],
    }
    np.testing.assert_array_equal(
        sim.agents['agent0'].initial_position,
        np.array([0, 0])
    )
    np.testing.assert_array_equal(
        sim.agents['agent1'].initial_position,
        np.array([0, 1])
    )
    np.testing.assert_array_equal(
        sim.agents['agent2'].initial_position,
        np.array([1, 0])
    )
    np.testing.assert_array_equal(
        sim.agents['agent3'].initial_position,
        np.array([1, 1])
    )
    assert next(iter(sim.grid[0, 0].values())) == agents[0]
    assert next(iter(sim.grid[0, 1].values())) == agents[1]
    assert next(iter(sim.grid[1, 0].values())) == agents[2]
    assert next(iter(sim.grid[1, 1].values())) == agents[3]

    sim.reset()
    assert next(iter(sim.grid[0, 0].values())) == agents[0]
    assert next(iter(sim.grid[0, 1].values())) == agents[1]
    assert next(iter(sim.grid[1, 0].values())) == agents[2]
    assert next(iter(sim.grid[1, 1].values())) == agents[3]

    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim_from_grid(grid._internal)

    with pytest.raises(AssertionError):
        agents[1].initial_position = np.array([1, 0])
        agents[2].initial_position = np.array([0, 1])
        MultiAgentGridSim.build_sim_from_grid(grid)


def test_build_sim_from_array():
    array = np.array([
        ['A', '.', 'B', '0', ''],
        ['B', '_', '', 'C', 'A']
    ])
    obj_registry = {
        'A': lambda n: GridWorldAgent(
            id=f'A-class-barrier{n}',
            encoding=1,
        ),
        'B': lambda n: GridWorldAgent(
            id=f'B-class-barrier{n}',
            encoding=2,
        ),
        'C': lambda n: GridWorldAgent(
            id=f'C-class-barrier{n}',
            encoding=3,
        ),
    }
    sim = MultiAgentGridSim.build_sim_from_array(array, obj_registry)
    assert sim.grid.rows == 2
    assert sim.grid.cols == 5
    np.testing.assert_array_equal(sim.grid._internal, np.empty((2, 5), dtype=object))

    sim.reset()
    assert 'A-class-barrier0' in sim.grid[0, 0]
    assert sim.grid[0, 1] == {}
    assert 'B-class-barrier1' in sim.grid[0, 2]
    assert sim.grid[0, 3] == {}
    assert sim.grid[0, 4] == {}
    assert 'B-class-barrier2' in sim.grid[1, 0]
    assert sim.grid[1, 1] == {}
    assert sim.grid[1, 2] == {}
    assert 'C-class-barrier3' in sim.grid[1, 3]
    assert 'A-class-barrier4' in sim.grid[1, 4]

    assert len(sim.agents) == 5
    assert sim.agents['A-class-barrier0'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier0'].initial_position,
        np.array([0, 0])    
    )
    assert sim.agents['B-class-barrier1'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier1'].initial_position,
        np.array([0, 2])    
    )
    assert sim.agents['B-class-barrier2'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier2'].initial_position,
        np.array([1, 0])    
    )
    assert sim.agents['C-class-barrier3'].encoding == 3
    np.testing.assert_array_equal(
        sim.agents['C-class-barrier3'].initial_position,
        np.array([1, 3])    
    )
    assert sim.agents['A-class-barrier4'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier4'].initial_position,
        np.array([1, 4])    
    )

    # Testin what happens when one of the keys is not in the registry
    del obj_registry['C']
    sim = MultiAgentGridSim.build_sim_from_array(array, obj_registry)
    assert sim.grid.rows == 2
    assert sim.grid.cols == 5

    sim.reset()
    assert 'A-class-barrier0' in sim.grid[0, 0]
    assert sim.grid[0, 1] == {}
    assert 'B-class-barrier1' in sim.grid[0, 2]
    assert sim.grid[0, 3] == {}
    assert sim.grid[0, 4] == {}
    assert 'B-class-barrier2' in sim.grid[1, 0]
    assert sim.grid[1, 1] == {}
    assert sim.grid[1, 2] == {}
    assert sim.grid[1, 3] == {}
    assert 'A-class-barrier3' in sim.grid[1, 4]

    assert len(sim.agents) == 4
    assert 'C-class-barrier3' not in sim.agents
    assert sim.agents['A-class-barrier0'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier0'].initial_position,
        np.array([0, 0])    
    )
    assert sim.agents['B-class-barrier1'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier1'].initial_position,
        np.array([0, 2])    
    )
    assert sim.agents['B-class-barrier2'].encoding == 2
    np.testing.assert_array_equal(
        sim.agents['B-class-barrier2'].initial_position,
        np.array([1, 0])    
    )
    assert sim.agents['A-class-barrier3'].encoding == 1
    np.testing.assert_array_equal(
        sim.agents['A-class-barrier3'].initial_position,
        np.array([1, 4])    
    )

    # Bad array
    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim_from_array(obj_registry, obj_registry)
    # Bad Object Registry
    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim_from_array(array, array)
    # Using reserved key
    with pytest.raises(AssertionError):
        obj_registry.update({
            '_': lambda n: GridWorldAgent(
                id=f'invalid_underscore!',
                encoding=0,
            ),
        })
        MultiAgentGridSim.build_sim_from_array(array, obj_registry)
