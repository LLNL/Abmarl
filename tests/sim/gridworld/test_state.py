
import numpy as np

from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.state import PositionState, MazePlacementState, HealthState, \
    TargetBarriersFreePlacementState, StateBaseComponent, AmmoState, OrientationState
from abmarl.sim.gridworld.agent import HealthAgent, GridWorldAgent, AmmoAgent, OrientationAgent
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


def test_position_state_no_overlap_at_reset():
    grid = Grid(3, 3, overlapping={1: {1}})
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1),
        'agent1': GridWorldAgent(id='agent1', encoding=1),
        'agent2': GridWorldAgent(id='agent2', encoding=1),
        'agent3': GridWorldAgent(id='agent3', encoding=1, initial_position=np.array([2, 2])),
        'agent4': GridWorldAgent(id='agent4', encoding=1),
        'agent5': GridWorldAgent(id='agent5', encoding=1),
        'agent6': GridWorldAgent(id='agent6', encoding=1),
        'agent7': GridWorldAgent(id='agent7', encoding=1),
        'agent8': GridWorldAgent(id='agent8', encoding=1, initial_position=np.array([0, 0])),
        'agent9': GridWorldAgent(id='agent9', encoding=1, initial_position=np.array([0, 0])),
    }
    position_state = PositionState(grid=grid, agents=agents, no_overlap_at_reset=True)
    position_state.reset()

    assert grid[0, 0] == {'agent8': agents['agent8'], 'agent9': agents['agent9']}
    assert grid[2, 2] == {'agent3': agents['agent3']}
    assert len(grid[0, 1]) == 1
    assert len(grid[0, 2]) == 1
    assert len(grid[1, 0]) == 1
    assert len(grid[1, 1]) == 1
    assert len(grid[1, 2]) == 1
    assert len(grid[2, 0]) == 1
    assert len(grid[2, 1]) == 1


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


def test_ammo_state():
    grid = Grid(3, 3)
    agents = {
        'agent0': AmmoAgent(id='agent0', encoding=1, initial_ammo=40),
        'agent1': AmmoAgent(id='agent1', encoding=1, initial_ammo=-2),
        'agent2': AmmoAgent(id='agent2', encoding=1, initial_ammo=13)
    }

    ammo_state = AmmoState(agents=agents, grid=grid)
    assert isinstance(ammo_state, StateBaseComponent)
    ammo_state.reset()

    assert agents['agent0'].ammo == 40
    assert agents['agent1'].ammo == 0
    assert agents['agent2'].ammo == 13

    agents['agent0'].ammo -= 16
    agents['agent1'].ammo += 7
    agents['agent2'].ammo -= 15
    assert agents['agent0'].ammo == 24
    assert agents['agent1'].ammo == 7
    assert agents['agent2'].ammo == 0

    agents['agent0'].ammo += 5
    assert agents['agent0'].ammo == 29


def test_maze_placement_state():
    target_agent = GridWorldAgent(id='target', encoding=1)
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
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3}
    )
    assert isinstance(state, PositionState)
    assert state.target_agent == target_agent
    assert state.barrier_encodings == {2}
    assert state.free_encodings == {1, 3}
    assert not state.cluster_barriers
    assert not state.scatter_free_agents

    state.reset()
    # No overlap between barriers and free
    assert not {*state.ravelled_positions_available[1]} & {*state.ravelled_positions_available[2]}
    assert not {*state.ravelled_positions_available[3]} & {*state.ravelled_positions_available[2]}
    # Target encoding not available to 1 but is avaiable to 3
    assert np.ravel_multi_index(
        target_agent.position, (5, 8)
    ) not in state.ravelled_positions_available[1]
    assert np.ravel_multi_index(
        target_agent.position, (5, 8)
    ) in state.ravelled_positions_available[3]
    # Free encodings available to target and to other free encodings
    for agent in free_agents.values():
        assert np.ravel_multi_index(
            agent.position, (5, 8)
        ) in state.ravelled_positions_available[3]
        if not np.array_equal(
                agent.position,
                state.target_agent.position
        ):
            # Free agent was placed at target, position won't be available to the
            # target
            assert np.ravel_multi_index(
                agent.position, (5, 8)
            ) in state.ravelled_positions_available[1]
    # None of the agents should have been given an initial position
    for agent in agents.values():
        assert not agent.initial_position


def test_maze_placement_state_target_by_id():
    target_agent = GridWorldAgent(id='target', encoding=1)
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
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent='target',
        barrier_encodings={2},
        free_encodings={1, 3}
    )
    assert isinstance(state, PositionState)
    assert state.target_agent == target_agent

    with pytest.raises(AssertionError):
        state = MazePlacementState(
            grid=grid,
            agents=agents,
            target_agent='target_agent',
            barrier_encodings={2},
            free_encodings={1, 3}
        )


def test_maze_placement_target_has_ip():
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
    agents = {
        'ip_agent': ip_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent=ip_agent,
        barrier_encodings={2},
        free_encodings={1, 3}
    )
    assert state.target_agent == ip_agent
    state.reset()
    np.testing.assert_array_equal(
        state.target_agent.position,
        state.target_agent.initial_position
    )


def test_maze_placement_nonoverlapping_ip_with_target():
    target_agent = GridWorldAgent(id='target', encoding=1)
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
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    target_agent.initial_position = ip_agent.initial_position
    agents = {
        'target': target_agent,
        'ip_agent': ip_agent,
        **barrier_agents,
        **free_agents
    }
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3}
    )
    with pytest.raises(AssertionError):
        state.reset()


def test_maze_placement_failures():
    target_agent = GridWorldAgent(id='target', encoding=1)
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
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    with pytest.raises(AssertionError):
        # Fails because there is no target agent
        MazePlacementState(
            grid=grid,
            agents=agents,
            barrier_encodings={2},
            free_encodings={1, 3}
        )

    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    with pytest.raises(AssertionError):
        # Fails because target agent not in the simulation
        MazePlacementState(
            grid=grid,
            agents=agents,
            target_agent=ip_agent,
            barrier_encodings={2},
            free_encodings={1, 3}
        )

    with pytest.raises(AssertionError):
        # Fails because barrier is list
        MazePlacementState(
            grid=grid,
            agents=agents,
            target_agent=target_agent,
            barrier_encodings=[2],
            free_encodings={1, 3}
        )

    with pytest.raises(AssertionError):
        # Fails because free is list
        MazePlacementState(
            grid=grid,
            agents=agents,
            target_agent=target_agent,
            barrier_encodings={2},
            free_encodings=[1, 3]
        )

    with pytest.raises(AssertionError):
        # Fails becuase some agents are neither barrier nor free
        state = MazePlacementState(
            grid=grid,
            agents=agents,
            target_agent=target_agent,
            barrier_encodings={2},
            free_encodings={1}
        )
        state.reset()


def test_maze_placement_state_no_barriers_no_free():
    target_agent = GridWorldAgent(id='target', encoding=1)
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
    agents = {
        'target': target_agent,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        free_encodings={1, 3}
    )
    assert isinstance(state, PositionState)
    assert state.target_agent == target_agent
    assert state.barrier_encodings == set()
    assert state.free_encodings == {1, 3}

    state.reset()


    barrier_agents = {
        f'barrier_agent{i}': GridWorldAgent(
            id=f'barrier_agent{i}',
            encoding=2
        ) for i in range(5)
    }
    agents = {
        'target': GridWorldAgent(id='target', encoding=2),
        **barrier_agents,
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent=agents['target'],
        barrier_encodings={2},
    )
    assert isinstance(state, PositionState)
    assert state.barrier_encodings == {2}
    assert state.free_encodings == set()

    state.reset()


def test_maze_placement_state_clustering_and_scattering():
    np.random.seed(24)
    target_agent = GridWorldAgent(id='target', encoding=1, initial_position=np.array([3, 2]))
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
        ) for i in range(5)
    }
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3},
        cluster_barriers=True,
        scatter_free_agents=True
    )
    assert state.cluster_barriers
    assert state.scatter_free_agents

    # Barrier are close to target, free agents are far from it
    # All the free agents start on the same cell
    state.reset()
    np.testing.assert_array_equal(
        target_agent.position,
        np.array([3, 2])
    )
    for barrier in barrier_agents.values():
        assert max(abs(target_agent.position - barrier.position)) <= 2

    for free_agent in free_agents.values():
        np.testing.assert_array_equal(
            free_agent.position,
            np.array([1, 7])
        )


def test_maze_placement_state_clustering_and_scattering_no_overlap_at_reset():
    np.random.seed(24)
    target_agent = GridWorldAgent(id='target', encoding=1, initial_position=np.array([3, 2]))
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
        ) for i in range(5)
    }
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        no_overlap_at_reset=True,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3},
        cluster_barriers=True,
        scatter_free_agents=True
    )
    # Barrier are close to target, free agents are far from it
    # All agents start on different cells
    state.reset()
    np.testing.assert_array_equal(
        target_agent.position,
        np.array([3, 2])
    )
    for r in range(5):
        for c in range(8):
            assert len(grid[r, c]) <= 1


def test_maze_placement_state_too_many_agents():
    target_agent = GridWorldAgent(id='target', encoding=1)
    barrier_agents = {
        f'barrier_agent{i}': GridWorldAgent(
            id=f'barrier_agent{i}',
            encoding=2
        ) for i in range(10)
    }
    free_agents = {
        f'free_agent{i}': GridWorldAgent(
            id=f'free_agent{i}',
            encoding=3
        ) for i in range(3)
    }
    agents = {
        'target': target_agent,
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
    with pytest.raises(RuntimeError):
        # Fails because cannot find cell
        state.reset()


    target_agent = GridWorldAgent(id='target', encoding=1)
    free_agents = {
        f'free_agent{i}': GridWorldAgent(
            id=f'free_agent{i}',
            encoding=3
        ) for i in range(20)
    }
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(4, 4)
    state = MazePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3}
    )
    with pytest.raises(RuntimeError):
        # Fails because cannot find cell
        state.reset()


def test_target_barrier_free_placement_state():
    target_agent = GridWorldAgent(id='target', encoding=1, initial_position=np.array([3, 2]))
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
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = TargetBarriersFreePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3}
    )
    assert isinstance(state, PositionState)
    assert state.target_agent == target_agent
    assert state.barrier_encodings == {2}
    assert state.free_encodings == {1, 3}
    assert not state.cluster_barriers
    assert not state.scatter_free_agents

    state.reset()
    np.testing.assert_array_equal(
        state.target_agent.position,
        state.target_agent.initial_position
    )


def test_target_barrier_free_placement_state_target_by_id():
    target_agent = GridWorldAgent(id='target', encoding=1, initial_position=np.array([3, 2]))
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
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = TargetBarriersFreePlacementState(
        grid=grid,
        agents=agents,
        target_agent='target',
        barrier_encodings={2},
        free_encodings={1, 3}
    )
    assert isinstance(state, PositionState)
    assert state.target_agent == target_agent

    with pytest.raises(AssertionError):
        state = TargetBarriersFreePlacementState(
            grid=grid,
            agents=agents,
            target_agent='target_agent',
            barrier_encodings={2},
            free_encodings={1, 3}
        )


def test_target_barrier_free_placement_failures():
    target_agent = GridWorldAgent(id='target', encoding=1)
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
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    with pytest.raises(AssertionError):
        # Fails because there is no target agent
        TargetBarriersFreePlacementState(
            grid=grid,
            agents=agents,
            barrier_encodings={2},
            free_encodings={1, 3}
        )

    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    with pytest.raises(AssertionError):
        # Fails because target agent not in the simulation
        TargetBarriersFreePlacementState(
            grid=grid,
            agents=agents,
            target_agent=ip_agent,
            barrier_encodings={2},
            free_encodings={1, 3}
        )

    with pytest.raises(AssertionError):
        # Fails because barrier is list
        TargetBarriersFreePlacementState(
            grid=grid,
            agents=agents,
            target_agent=target_agent,
            barrier_encodings=[2],
            free_encodings={1, 3}
        )

    with pytest.raises(AssertionError):
        # Fails because free is list
        TargetBarriersFreePlacementState(
            grid=grid,
            agents=agents,
            target_agent=target_agent,
            barrier_encodings={2},
            free_encodings=[1, 3]
        )

    with pytest.raises(AssertionError):
        # Fails because some agents are neither barrier nor free
        state = TargetBarriersFreePlacementState(
            grid=grid,
            agents=agents,
            target_agent=target_agent,
            barrier_encodings={2},
            free_encodings={1}
        )
        state.reset()


def test_target_barrier_free_placement_state_no_barriers_no_free():
    target_agent = GridWorldAgent(id='target', encoding=1)
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
    agents = {
        'target': target_agent,
        **free_agents
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = TargetBarriersFreePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        free_encodings={1, 3}
    )
    assert isinstance(state, PositionState)
    assert state.target_agent == target_agent
    assert state.barrier_encodings == set()
    assert state.free_encodings == {1, 3}

    state.reset()


    barrier_agents = {
        f'barrier_agent{i}': GridWorldAgent(
            id=f'barrier_agent{i}',
            encoding=2
        ) for i in range(5)
    }
    agents = {
        'target': GridWorldAgent(id='target', encoding=2),
        **barrier_agents,
    }
    grid = Grid(5, 8, overlapping={1: {3}, 3: {3}})
    state = TargetBarriersFreePlacementState(
        grid=grid,
        agents=agents,
        target_agent=agents['target'],
        barrier_encodings={2},
    )
    assert isinstance(state, PositionState)
    assert state.barrier_encodings == {2}
    assert state.free_encodings == set()

    state.reset()


def test_target_barriers_free_placement_state_clustering_and_scattering():
    np.random.seed(24)
    target_agent = GridWorldAgent(id='target', encoding=1, initial_position=np.array([3, 2]))
    barrier_agents = {
        f'barrier_agent{i}': GridWorldAgent(
            id=f'barrier_agent{i}',
            encoding=2
        ) for i in range(24)
    }
    free_agents = {
        f'free_agent{i}': GridWorldAgent(
            id=f'free_agent{i}',
            encoding=3
        ) for i in range(5)
    }
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(6, 8, overlapping={1: {3}, 3: {3}})
    state = TargetBarriersFreePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3},
        cluster_barriers=True,
        scatter_free_agents=True
    )
    assert state.cluster_barriers
    assert state.scatter_free_agents

    # Barrier are close to target, free agents are far from it
    # All the free agents start on the same cell
    state.reset()
    np.testing.assert_array_equal(
        target_agent.position,
        np.array([3, 2])
    )
    for barrier in barrier_agents.values():
        assert max(abs(target_agent.position - barrier.position)) <= 2

    for free_agent in free_agents.values():
        np.testing.assert_array_equal(
            free_agent.position,
            np.array([0, 7])
        )


def test_target_barriers_free_placement_state_clustering_and_scattering_no_overlap_at_reset():
    np.random.seed(24)
    target_agent = GridWorldAgent(id='target', encoding=1, initial_position=np.array([3, 2]))
    barrier_agents = {
        f'barrier_agent{i}': GridWorldAgent(
            id=f'barrier_agent{i}',
            encoding=2
        ) for i in range(24)
    }
    free_agents = {
        f'free_agent{i}': GridWorldAgent(
            id=f'free_agent{i}',
            encoding=3
        ) for i in range(5)
    }
    agents = {
        'target': target_agent,
        **barrier_agents,
        **free_agents
    }
    grid = Grid(6, 8, overlapping={1: {3}, 3: {3}})
    state = TargetBarriersFreePlacementState(
        grid=grid,
        agents=agents,
        no_overlap_at_reset=True,
        target_agent=target_agent,
        barrier_encodings={2},
        free_encodings={1, 3},
        cluster_barriers=True,
        scatter_free_agents=True
    )
    # Barrier are close to target, free agents are far from it
    # All agents start on different cells
    state.reset()
    np.testing.assert_array_equal(
        target_agent.position,
        np.array([3, 2])
    )
    for r in range(5):
        for c in range(8):
            assert len(grid[r, c]) <= 1
    for barrier in barrier_agents.values():
        assert max(abs(target_agent.position - barrier.position)) <= 2
    for free in free_agents.values():
        assert max(abs(target_agent.position - free.position)) > 2


def test_target_barrier_free_placement_state_random_order():
    target_agent = GridWorldAgent(id='target', encoding=1, initial_position=np.array([0, 0]))
    barrier_agents = {
        f'barrier_agent{i}': GridWorldAgent(
            id=f'barrier_agent{i}',
            encoding=2
        ) for i in range(3)
    }
    agents = {
        'target': target_agent,
        **barrier_agents,
    }
    grid = Grid(1, 4)
    state = TargetBarriersFreePlacementState(
        grid=grid,
        agents=agents,
        target_agent=target_agent,
        barrier_encodings={1, 2},
        cluster_barriers=True,
        randomize_placement_order=False
    )
    assert not state.randomize_placement_order

    state.reset()
    assert 'barrier_agent0' in grid[0, 1]
    assert len(grid[0, 1]) == 1

    np.random.seed(24)
    state.randomize_placement_order = True

    count_0_closest = 0
    for _ in range(10):
        state.reset()
        if 'barrier_agent0' in grid[0, 1]:
            count_0_closest += 1
    assert count_0_closest < 10


def test_orientation_state():
    agents = {
        f'agent_{o}': OrientationAgent(
            id=f'agent_{o}',
            encoding=o,
            initial_orientation=o
        ) for o in range(1, 5)
    }
    grid = Grid(1, 4)
    state = OrientationState(agents=agents, grid=grid)
    assert isinstance(state, OrientationState)
    assert isinstance(state, StateBaseComponent)

    state.reset()
    for agent in agents.values():
        assert agent.orientation == agent.initial_orientation
