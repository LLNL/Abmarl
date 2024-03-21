
import os

import numpy as np
import pytest

from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.agent import GridWorldAgent
from abmarl.examples.sim import MultiAgentGridSim


def test_build():
    agent = GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 0]))
    sim = MultiAgentGridSim.build_sim(
        3, 4,
        agents={'agent0': agent}
    )
    assert sim.agents == {'agent0': agent}
    assert isinstance(sim.grid, Grid)
    assert sim.grid.rows == 3
    assert sim.grid.cols == 4
    np.testing.assert_array_equal(sim.grid._internal, np.empty((3, 4), dtype=object))

    sim.reset()
    np.testing.assert_array_equal(
        sim.grid._internal,  np.array([
            [{'agent0': agent}, {}, {}, {}],
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
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 0])),
        'agent1': GridWorldAgent(id='agent1', encoding=1, initial_position=np.array([0, 1])),
        'agent2': GridWorldAgent(id='agent2', encoding=1, initial_position=np.array([1, 0])),
        'agent3': GridWorldAgent(id='agent3', encoding=1, initial_position=np.array([1, 1])),
    }
    grid.place(agents['agent0'], (0, 0))
    grid.place(agents['agent1'], (0, 1))
    grid.place(agents['agent2'], (1, 0))
    grid.place(agents['agent3'], (1, 1))

    sim = MultiAgentGridSim.build_sim_from_grid(grid)
    assert sim.grid.rows == 2
    assert sim.grid.cols == 2
    np.testing.assert_array_equal(sim.grid._internal, np.empty((2, 2), dtype=object))

    sim.reset()
    assert sim.agents == {
        'agent0': agents['agent0'],
        'agent1': agents['agent1'],
        'agent2': agents['agent2'],
        'agent3': agents['agent3'],
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
    assert next(iter(sim.grid[0, 0].values())) == agents['agent0']
    assert next(iter(sim.grid[0, 1].values())) == agents['agent1']
    assert next(iter(sim.grid[1, 0].values())) == agents['agent2']
    assert next(iter(sim.grid[1, 1].values())) == agents['agent3']

    with pytest.raises(AssertionError):
        # This fails because the grid must be a grid object, not an array
        MultiAgentGridSim.build_sim_from_grid(grid._internal)

    with pytest.raises(AssertionError):
        # This fails becaue the agents' initial positions must match their index
        # within the grid.
        agents['agent1'].initial_position = np.array([1, 0])
        agents['agent2'].initial_position = np.array([0, 1])
        MultiAgentGridSim.build_sim_from_grid(grid)


def test_build_from_grid_with_extra_agents():
    grid = Grid(2, 2)
    grid.reset()
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1, initial_position=np.array([0, 0])),
        'agent1': GridWorldAgent(id='agent1', encoding=1, initial_position=np.array([0, 1])),
        'agent2': GridWorldAgent(id='agent2', encoding=1, initial_position=np.array([1, 0])),
    }
    grid.place(agents['agent0'], (0, 0))
    grid.place(agents['agent1'], (0, 1))
    grid.place(agents['agent2'], (1, 0))

    # Agent 0 is already in the grid, so the builder should ignore agent0 in the
    # extra agents. Agents 3 and 4 should be there because they are overlappable.
    # Agent5 should be there because it can take the last empty spot on the map.
    extra_agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=2, initial_position=np.array([0, 1])),
        'agent3': GridWorldAgent(id='agent3', encoding=3, initial_position=np.array([0, 1])),
        'agent4': GridWorldAgent(id='agent4', encoding=4, initial_position=np.array([1, 0])),
        'agent5': GridWorldAgent(id='agent5', encoding=5),
    }

    sim = MultiAgentGridSim.build_sim_from_grid(
        grid,
        extra_agents=extra_agents,
        overlapping={1: {3, 4}, 3: {1}, 4: {1}}
    )
    sim.reset()
    assert sim.agents == {
        'agent0': agents['agent0'], # We use the agent already in the grid, not from extra agents.
        'agent1': agents['agent1'],
        'agent2': agents['agent2'],
        'agent3': extra_agents['agent3'],
        'agent4': extra_agents['agent4'],
        'agent5': extra_agents['agent5'],
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
        np.array([0, 1])
    )
    np.testing.assert_array_equal(
        sim.agents['agent4'].initial_position,
        np.array([1, 0])
    )
    assert sim.agents['agent5'].initial_position is None

    assert next(iter(sim.grid[0, 0].values())) == agents['agent0']
    assert agents['agent1'] in sim.grid[0, 1].values()
    assert agents['agent2'] in sim.grid[1, 0].values()
    assert extra_agents['agent3'] in sim.grid[0, 1].values()
    assert extra_agents['agent4'] in sim.grid[1, 0].values()
    assert next(iter(sim.grid[1, 1].values())) == extra_agents['agent5']

    with pytest.raises(AssertionError):
        # This fails because extra agents must be a dict
        MultiAgentGridSim.build_sim_from_grid(grid, extra_agents=[])

    with pytest.raises(AssertionError):
        # This fails because the contents of the extra_agents dict is wrong.
        MultiAgentGridSim.build_sim_from_grid(grid, extra_agents={0: 1})

    sim2 = MultiAgentGridSim.build_sim_from_grid(
        grid,
        extra_agents=extra_agents
    )
    with pytest.raises(AssertionError):
        # This fails because the agents are not allowed to overlap
        sim2.reset()


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
                id='invalid_underscore!',
                encoding=0,
            ),
        })
        MultiAgentGridSim.build_sim_from_array(array, obj_registry)


def test_build_sim_from_array_with_extra_agents():
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
    # B-class-barrier2 exists in the array, so the builder should not use the one
    # in extra_agents
    # extra_agent0 and 1 will exist because they can overlap
    # extra_agent2 will exist because it occupies an empty space
    extra_agents = {
        'B-class-barrier2': GridWorldAgent(
            id='B-class-barrier2',
            encoding=4,
            initial_position=np.array([1, 0])
        ),
        'extra_agent0': GridWorldAgent(
            id='extra_agent0',
            encoding=5,
            initial_position=np.array([0, 0])
        ),
        'extra_agent1': GridWorldAgent(
            id='extra_agent1',
            encoding=5,
            initial_position=np.array([0, 0])
        ),
        'extra_agent2': GridWorldAgent(
            id='extra_agent2',
            encoding=6,
            initial_position=np.array([0, 4])
        )
    }
    sim = MultiAgentGridSim.build_sim_from_array(
        array,
        obj_registry,
        extra_agents=extra_agents,
        overlapping={1: {5}, 5: {1, 5}}
    )
    sim.reset()
    assert 'A-class-barrier0' in sim.grid[0, 0]
    assert 'extra_agent0' in sim.grid[0, 0]
    assert 'extra_agent1' in sim.grid[0, 0]
    assert sim.grid[0, 1] == {}
    assert 'B-class-barrier1' in sim.grid[0, 2]
    assert sim.grid[0, 3] == {}
    assert next(iter(sim.grid[0, 4].values())) == extra_agents['extra_agent2']
    assert 'B-class-barrier2' in sim.grid[1, 0]
    assert next(iter(sim.grid[1, 0].values())).encoding == 2
    assert sim.grid[1, 1] == {}
    assert sim.grid[1, 2] == {}
    assert 'C-class-barrier3' in sim.grid[1, 3]
    assert 'A-class-barrier4' in sim.grid[1, 4]

    assert len(sim.agents) == 8
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
    assert sim.agents['extra_agent0'].encoding == 5
    np.testing.assert_array_equal(
        sim.agents['extra_agent0'].initial_position,
        np.array([0, 0])
    )
    assert sim.agents['extra_agent1'].encoding == 5
    np.testing.assert_array_equal(
        sim.agents['extra_agent1'].initial_position,
        np.array([0, 0])
    )
    assert sim.agents['extra_agent2'].encoding == 6
    np.testing.assert_array_equal(
        sim.agents['extra_agent2'].initial_position,
        np.array([0, 4])
    )

    with pytest.raises(AssertionError):
        # This fails because extra agents must be a dict
        MultiAgentGridSim.build_sim_from_array(array, obj_registry, extra_agents=[])

    with pytest.raises(AssertionError):
        # This fails because the contents of the extra_agents dict is wrong.
        MultiAgentGridSim.build_sim_from_array(array, obj_registry, extra_agents={0: 1})

    sim2 = MultiAgentGridSim.build_sim_from_array(
        array,
        obj_registry,
        extra_agents=extra_agents,
        overlapping={1: {5}, 5: {1}}
    )
    with pytest.raises(AssertionError):
        # This fails because 5 cannot overlap with 5
        sim2.reset()


def test_build_sim_from_file():
    file_name = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'grid_file.txt'
    )
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
    sim = MultiAgentGridSim.build_sim_from_file(file_name, obj_registry)
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
    sim = MultiAgentGridSim.build_sim_from_file(file_name, obj_registry)
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
        MultiAgentGridSim.build_sim_from_file(obj_registry, obj_registry)
    # Bad Object Registry
    with pytest.raises(AssertionError):
        MultiAgentGridSim.build_sim_from_file(file_name, file_name)
    # Using reserved key
    with pytest.raises(AssertionError):
        obj_registry.update({
            '_': lambda n: GridWorldAgent(
                id='invalid_underscore!',
                encoding=0,
            ),
        })
        MultiAgentGridSim.build_sim_from_file(file_name, obj_registry)


def test_build_sim_from_file_with_extra_agents():
    file_name = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'grid_file.txt'
    )
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
    # B-class-barrier2 exists in the file, so the builder should not use the one
    # in extra_agents
    # extra_agent0 and 1 will exist because they can overlap
    # extra_agent2 will exist because it occupies an empty space
    extra_agents = {
        'B-class-barrier2': GridWorldAgent(
            id='B-class-barrier2',
            encoding=4,
            initial_position=np.array([1, 0])
        ),
        'extra_agent0': GridWorldAgent(
            id='extra_agent0',
            encoding=5,
            initial_position=np.array([0, 0])
        ),
        'extra_agent1': GridWorldAgent(
            id='extra_agent1',
            encoding=5,
            initial_position=np.array([0, 0])
        ),
        'extra_agent2': GridWorldAgent(
            id='extra_agent2',
            encoding=6,
            initial_position=np.array([0, 4])
        )
    }
    sim = MultiAgentGridSim.build_sim_from_file(
        file_name,
        obj_registry,
        extra_agents=extra_agents,
        overlapping={1: {5}, 5: {1, 5}}
    )
    sim.reset()
    assert 'A-class-barrier0' in sim.grid[0, 0]
    assert 'extra_agent0' in sim.grid[0, 0]
    assert 'extra_agent1' in sim.grid[0, 0]
    assert sim.grid[0, 1] == {}
    assert 'B-class-barrier1' in sim.grid[0, 2]
    assert sim.grid[0, 3] == {}
    assert next(iter(sim.grid[0, 4].values())) == extra_agents['extra_agent2']
    assert 'B-class-barrier2' in sim.grid[1, 0]
    assert next(iter(sim.grid[1, 0].values())).encoding == 2
    assert sim.grid[1, 1] == {}
    assert sim.grid[1, 2] == {}
    assert 'C-class-barrier3' in sim.grid[1, 3]
    assert 'A-class-barrier4' in sim.grid[1, 4]

    assert len(sim.agents) == 8
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
    assert sim.agents['extra_agent0'].encoding == 5
    np.testing.assert_array_equal(
        sim.agents['extra_agent0'].initial_position,
        np.array([0, 0])
    )
    assert sim.agents['extra_agent1'].encoding == 5
    np.testing.assert_array_equal(
        sim.agents['extra_agent1'].initial_position,
        np.array([0, 0])
    )
    assert sim.agents['extra_agent2'].encoding == 6
    np.testing.assert_array_equal(
        sim.agents['extra_agent2'].initial_position,
        np.array([0, 4])
    )

    with pytest.raises(AssertionError):
        # This fails because extra agents must be a dict
        MultiAgentGridSim.build_sim_from_file(file_name, obj_registry, extra_agents=[])

    with pytest.raises(AssertionError):
        # This fails because the contents of the extra_agents dict is wrong.
        MultiAgentGridSim.build_sim_from_file(file_name, obj_registry, extra_agents={0: 1})

    sim2 = MultiAgentGridSim.build_sim_from_file(
        file_name,
        obj_registry,
        extra_agents=extra_agents,
        overlapping={1: {5}, 5: {1}}
    )
    with pytest.raises(AssertionError):
        # This fails because 5 cannot overlap with 5
        sim2.reset()
