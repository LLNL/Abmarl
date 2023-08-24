
import numpy as np

from abmarl.tools import Box
from abmarl.sim.agent_based_simulation import ObservingAgent
from abmarl.sim.gridworld.observer import ObserverBaseComponent, AbsoluteEncodingObserver, \
    PositionCenteredEncodingObserver, StackedPositionCenteredEncodingObserver, \
    AbsolutePositionObserver, AmmoObserver
from abmarl.sim.gridworld.agent import GridObservingAgent, GridWorldAgent, MovingAgent, \
    AmmoAgent, AmmoObservingAgent
from abmarl.sim.gridworld.state import PositionState, AmmoState
from abmarl.sim.gridworld.grid import Grid


def test_ammo_observer():
    grid = Grid(3, 3)
    agents = {
        'agent0': AmmoAgent(id='agent0', encoding=1, initial_ammo=10),
        'agent1': AmmoObservingAgent(id='agent1', encoding=1, initial_ammo=-3),
        'agent2': AmmoObservingAgent(id='agent2', encoding=1, initial_ammo=14),
        'agent3': AmmoObservingAgent(id='agent3', encoding=1, initial_ammo=12),
    }
    state = AmmoState(grid=grid, agents=agents)
    observer = AmmoObserver(grid=grid, agents=agents)
    assert isinstance(observer, ObserverBaseComponent)
    state.reset()

    assert observer.get_obs(agents['agent1'])['ammo'] == agents['agent1'].ammo
    assert observer.get_obs(agents['agent2'])['ammo'] == agents['agent2'].ammo
    assert observer.get_obs(agents['agent3'])['ammo'] == agents['agent3'].ammo

    agents['agent0'].ammo -= 16
    agents['agent1'].ammo += 7
    agents['agent2'].ammo -= 15
    assert observer.get_obs(agents['agent1'])['ammo'] == agents['agent1'].ammo
    assert observer.get_obs(agents['agent2'])['ammo'] == agents['agent2'].ammo
    assert observer.get_obs(agents['agent3'])['ammo'] == agents['agent3'].ammo

    assert not observer.get_obs(agents['agent0'])


def test_absolute_encoding_observer():
    np.random.seed(24)
    grid = Grid(5, 5, overlapping={1: {6}, 6: {1}})
    agents = {
        'agent0': GridObservingAgent(
            id='agent0', encoding=1, view_range=2, initial_position=np.array([2, 2])
        ),
        'agent1': GridObservingAgent(
            id='agent1', encoding=2, view_range=1, initial_position=np.array([0, 0])
        ),
        'agent2': GridObservingAgent(
            id='agent2', encoding=3, view_range=4, initial_position=np.array([4, 4])
        ),
        'agent3': GridWorldAgent(
            id='agent3', encoding=5, initial_position=np.array([3, 3])
        ),
        'agent4': GridWorldAgent(
            id='agent4', encoding=4, initial_position=np.array([1, 1])
        ),
        'agent5': GridWorldAgent(
            id='agent5', encoding=6, initial_position=np.array([2, 1])
        ),
        'agent6': GridWorldAgent(
            id='agent6', encoding=6, initial_position=np.array([2, 2])
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    observer = AbsoluteEncodingObserver(agents=agents, grid=grid)
    assert isinstance(observer, ObserverBaseComponent)
    position_state.reset()

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['absolute_encoding'],
        np.array([
            [ 2,  0,  0,  0,  0],
            [ 0,  4,  0,  0,  0],
            [ 0,  6, -1,  0,  0],
            [ 0,  0,  0,  5,  0],
            [ 0,  0,  0,  0,  3]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['absolute_encoding'],
        np.array([
            [-1,  0, -2, -2, -2],
            [ 0,  4, -2, -2, -2],
            [-2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['absolute_encoding'],
        np.array([
            [ 2,  0,  0,  0,  0],
            [ 0,  4,  0,  0,  0],
            [ 0,  6,  1,  0,  0],
            [ 0,  0,  0,  5,  0],
            [ 0,  0,  0,  0, -1]
        ])
    )


def test_absolute_encoding_observer_blocking():
    np.random.seed(24)
    grid = Grid(5, 5, overlapping={1: {6}, 6: {1}})
    agents = {
        'agent0': GridObservingAgent(
            id='agent0', encoding=1, view_range=2, initial_position=np.array([2, 2])
        ),
        'agent1': GridObservingAgent(
            id='agent1', encoding=2, view_range=1, initial_position=np.array([0, 0])
        ),
        'agent2': GridObservingAgent(
            id='agent2', encoding=3, view_range=4, initial_position=np.array([4, 4])
        ),
        'agent3': GridWorldAgent(
            id='agent3', encoding=5, initial_position=np.array([3, 3]), blocking=True
        ),
        'agent4': GridWorldAgent(
            id='agent4', encoding=4, initial_position=np.array([1, 1]), blocking=True
        ),
        'agent5': GridWorldAgent(
            id='agent5', encoding=6, initial_position=np.array([2, 1]), blocking=True
        ),
        'agent6': GridWorldAgent(
            id='agent6', encoding=6, initial_position=np.array([2, 2])
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    observer = AbsoluteEncodingObserver(agents=agents, grid=grid)
    assert isinstance(observer, ObserverBaseComponent)
    position_state.reset()

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['absolute_encoding'],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  4,  0,  0,  0],
            [-2,  6, -1,  0,  0],
            [-2,  0,  0,  5, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['absolute_encoding'],
        np.array([
            [-1,  0, -2, -2, -2],
            [ 0,  4, -2, -2, -2],
            [-2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['absolute_encoding'],
        np.array([
            [-2, -2, -2,  0,  0],
            [-2, -2, -2,  0,  0],
            [-2, -2, -2, -2,  0],
            [ 0,  0, -2,  5,  0],
            [ 0,  0,  0,  0, -1]
        ])
    )


    agents['agent3'].active = False

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['absolute_encoding'],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  4,  0,  0,  0],
            [-2,  6, -1,  0,  0],
            [-2,  0,  0,  5,  0],
            [ 0,  0,  0,  0,  3]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['absolute_encoding'],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  4,  0,  0,  0],
            [-2,  6,  1,  0,  0],
            [ 0,  0,  0,  5,  0],
            [ 0,  0,  0,  0, -1]
        ])
    )


def test_single_grid_observer():
    grid = Grid(5, 5)
    agents = {
        'agent0': GridObservingAgent(
            id='agent0', encoding=1, view_range=2, initial_position=np.array([2, 2])
        ),
        'agent1': GridObservingAgent(
            id='agent1', encoding=2, view_range=1, initial_position=np.array([0, 0])
        ),
        'agent2': GridObservingAgent(
            id='agent2', encoding=3, view_range=4, initial_position=np.array([4, 4])
        ),
        'agent3': GridWorldAgent(id='agent3', encoding=5, initial_position=np.array([3, 3])),
        'agent4': GridWorldAgent(id='agent4', encoding=4, initial_position=np.array([1, 1])),
        'agent5': GridWorldAgent(id='agent5', encoding=6, initial_position=np.array([2, 1])),
    }

    position_state = PositionState(grid=grid, agents=agents)
    observer = PositionCenteredEncodingObserver(agents=agents, grid=grid)
    assert observer.key == 'position_centered_encoding'
    assert observer.supported_agent_type == GridObservingAgent
    assert isinstance(observer, ObserverBaseComponent)
    assert agents['agent0'].observation_space['position_centered_encoding'] == Box(
        -2, 6, (5, 5), int
    )
    assert agents['agent1'].observation_space['position_centered_encoding'] == Box(
        -2, 6, (3, 3), int
    )
    assert agents['agent2'].observation_space['position_centered_encoding'] == Box(
        -2, 6, (9, 9), int
    )

    agents['agent0'].finalize()
    assert agents['agent0'].null_observation.keys() == set(('position_centered_encoding',))
    np.testing.assert_array_equal(
        agents['agent0'].null_observation['position_centered_encoding'],
        -2 * np.ones((5, 5), dtype=int)
    )
    agents['agent1'].finalize()
    assert agents['agent1'].null_observation.keys() == set(('position_centered_encoding',))
    np.testing.assert_array_equal(
        agents['agent1'].null_observation['position_centered_encoding'],
        -2 * np.ones((3, 3), dtype=int)
    )
    agents['agent2'].finalize()
    assert agents['agent2'].null_observation.keys() == set(('position_centered_encoding',))
    np.testing.assert_array_equal(
        agents['agent2'].null_observation['position_centered_encoding'],
        -2 * np.ones((9, 9), dtype=int)
    )

    position_state.reset()
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['position_centered_encoding'],
        np.array([
            [2, 0, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 6, 1, 0, 0],
            [0, 0, 0, 5, 0],
            [0, 0, 0, 0, 3]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['position_centered_encoding'],
        np.array([
            [-1, -1, -1],
            [-1,  2,  0],
            [-1,  0,  4]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['position_centered_encoding'],
        np.array([
            [ 2,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  4,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  6,  1,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  5,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  3, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )


def test_single_grid_observer_blocking():
    grid = Grid(5, 5)
    agents = {
        'agent0': GridObservingAgent(
            id='agent0', encoding=1, view_range=2, initial_position=np.array([2, 2])
        ),
        'agent1': GridObservingAgent(
            id='agent1', encoding=2, view_range=1, initial_position=np.array([0, 0])
        ),
        'agent2': GridObservingAgent(
            id='agent2', encoding=3, view_range=4, initial_position=np.array([4, 4])
        ),
        'agent3': GridWorldAgent(
            id='agent3', encoding=5, initial_position=np.array([3, 3]), blocking=True
        ),
        'agent4': GridWorldAgent(
            id='agent4', encoding=4, initial_position=np.array([1, 1]), blocking=True
        ),
        'agent5': GridWorldAgent(
            id='agent5', encoding=6, initial_position=np.array([2, 1]), blocking=True
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    observer = PositionCenteredEncodingObserver(agents=agents, grid=grid)
    assert isinstance(observer, ObserverBaseComponent)
    position_state.reset()

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['position_centered_encoding'],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  4,  0,  0,  0],
            [-2,  6,  1,  0,  0],
            [-2,  0,  0,  5, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['position_centered_encoding'],
        np.array([
            [-1, -1, -1],
            [-1,  2,  0],
            [-1,  0,  4]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['position_centered_encoding'],
        np.array([
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2, -2,  0, -1, -1, -1, -1],
            [ 0,  0, -2,  5,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  3, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )


def test_multi_grid_observer():
    class HackAgent(GridObservingAgent, MovingAgent): pass

    agents = {
        'agent0': GridObservingAgent(
            id='agent0', encoding=1, view_range=2, initial_position=np.array([2, 2])
        ),
        'agent1': GridObservingAgent(
            id='agent1', encoding=2, view_range=1, initial_position=np.array([0, 0])
        ),
        'agent2': GridObservingAgent(
            id='agent2', encoding=3, view_range=4, initial_position=np.array([4, 4])
        ),
        'agent6': HackAgent(
            id='agent6', encoding=2, view_range=1, initial_position=np.array([4, 4]), move_range=1
        ),
        'agent7': HackAgent(
            id='agent7', encoding=3, view_range=4, initial_position=np.array([0, 0]), move_range=1
        ),
        'agent3': GridWorldAgent(
            id='agent3', encoding=5, initial_position=np.array([3, 3])
        ),
        'agent8': MovingAgent(
            id='agent8', encoding=5, initial_position=np.array([3, 3]), move_range=1
        ),
        'agent4': GridWorldAgent(
            id='agent4', encoding=4, initial_position=np.array([1, 1])
        ),
        'agent5': GridWorldAgent(
            id='agent5', encoding=6, initial_position=np.array([2, 1])
        ),
    }
    grid = Grid(5, 5, overlapping={2: {3}, 3: {2}, 5: {5}})

    position_state = PositionState(grid=grid, agents=agents)
    observer = StackedPositionCenteredEncodingObserver(agents=agents, grid=grid)
    assert observer.key == 'stacked_position_centered_encoding'
    assert observer.supported_agent_type == GridObservingAgent
    assert isinstance(observer, ObserverBaseComponent)
    assert observer.number_of_encodings == 6
    assert agents['agent0'].observation_space['stacked_position_centered_encoding'] == Box(
        -2, 9, (5, 5, 6), int
    )
    assert agents['agent1'].observation_space['stacked_position_centered_encoding'] == Box(
        -2, 9, (3, 3, 6), int
    )
    assert agents['agent2'].observation_space['stacked_position_centered_encoding'] == Box(
        -2, 9, (9, 9, 6), int
    )

    agents['agent0'].finalize()
    assert agents['agent0'].null_observation.keys() == set(('stacked_position_centered_encoding',))
    np.testing.assert_array_equal(
        agents['agent0'].null_observation['stacked_position_centered_encoding'],
        -2 * np.ones((5, 5, 6), dtype=int)
    )
    agents['agent1'].finalize()
    assert agents['agent1'].null_observation.keys() == set(('stacked_position_centered_encoding',))
    np.testing.assert_array_equal(
        agents['agent1'].null_observation['stacked_position_centered_encoding'],
        -2 * np.ones((3, 3, 6), dtype=int)
    )
    agents['agent2'].finalize()
    assert agents['agent2'].null_observation.keys() == set(('stacked_position_centered_encoding',))
    np.testing.assert_array_equal(
        agents['agent2'].null_observation['stacked_position_centered_encoding'],
        -2 * np.ones((9, 9, 6), dtype=int)
    )

    position_state.reset()

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 0],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 1],
        np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 2],
        np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 3],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 4],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 5],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,0],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,1],
        np.array([
            [-1, -1, -1],
            [-1,  1,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,2],
        np.array([
            [-1, -1, -1],
            [-1,  1,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,3],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,4],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,5],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,0],
        np.array([
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  1,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,1],
        np.array([
            [ 1,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,2],
        np.array([
            [ 1,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,3],
        np.array([
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  1,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,4],
        np.array([
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  2,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,5],
        np.array([
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  1,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )


def test_multi_grid_observer_blocking():
    class HackAgent(GridObservingAgent, MovingAgent): pass

    agents = {
        'agent0': GridObservingAgent(
            id='agent0', encoding=1, view_range=2, initial_position=np.array([2, 2])
        ),
        'agent1': GridObservingAgent(
            id='agent1', encoding=2, view_range=1, initial_position=np.array([0, 0])
        ),
        'agent2': GridObservingAgent(
            id='agent2', encoding=3, view_range=4, initial_position=np.array([4, 4])
        ),
        'agent6': HackAgent(
            id='agent6', encoding=2, view_range=1, initial_position=np.array([4, 4]), move_range=1
        ),
        'agent7': HackAgent(
            id='agent7', encoding=3, view_range=4, initial_position=np.array([0, 0]), move_range=1
        ),
        'agent3': GridWorldAgent(
            id='agent3', encoding=5, initial_position=np.array([3, 3]), blocking=True
        ),
        'agent8': MovingAgent(
            id='agent8', encoding=5, initial_position=np.array([3, 3]), move_range=1, blocking=True
        ),
        'agent4': GridWorldAgent(
            id='agent4', encoding=4, initial_position=np.array([1, 1]), blocking=True
        ),
        'agent5': GridWorldAgent(
            id='agent5', encoding=6, initial_position=np.array([2, 1]), blocking=True
        ),
    }
    grid = Grid(5, 5, overlapping={2: {3}, 3: {2}, 5: {5}})

    position_state = PositionState(grid=grid, agents=agents)
    observer = StackedPositionCenteredEncodingObserver(agents=agents, grid=grid)
    assert isinstance(observer, ObserverBaseComponent)
    position_state.reset()

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 0],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  1,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 1],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 2],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 3],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  1,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 4],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  2, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['stacked_position_centered_encoding'][:, :, 5],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  1,  0,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,0],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,1],
        np.array([
            [-1, -1, -1],
            [-1,  1,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,2],
        np.array([
            [-1, -1, -1],
            [-1,  1,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,3],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,4],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['stacked_position_centered_encoding'][:,:,5],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,0],
        np.array([
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2, -2,  0, -1, -1, -1, -1],
            [ 0,  0, -2,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,1],
        np.array([
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2, -2,  0, -1, -1, -1, -1],
            [ 0,  0, -2,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,2],
        np.array([
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2, -2,  0, -1, -1, -1, -1],
            [ 0,  0, -2,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,3],
        np.array([
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2, -2,  0, -1, -1, -1, -1],
            [ 0,  0, -2,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,4],
        np.array([
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2, -2,  0, -1, -1, -1, -1],
            [ 0,  0, -2,  2,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['stacked_position_centered_encoding'][:,:,5],
        np.array([
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2,  0,  0, -1, -1, -1, -1],
            [-2, -2, -2, -2,  0, -1, -1, -1, -1],
            [ 0,  0, -2,  0,  0, -1, -1, -1, -1],
            [ 0,  0,  0,  0,  0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    )


def test_observe_self():
    np.random.seed(24)
    class HackAgent(GridObservingAgent, MovingAgent): pass

    agents = {
        'agent0': GridObservingAgent(
            id='agent0', encoding=1, view_range=2, initial_position=np.array([2, 2])
        ),
        'agent1': GridObservingAgent(
            id='agent1', encoding=2, view_range=1, initial_position=np.array([0, 0])
        ),
        'agent2': HackAgent(
            id='agent2', encoding=2, view_range=1, initial_position=np.array([2, 2]), move_range=1
        ),
    }
    grid = Grid(5, 5, overlapping={1: {2}, 2: {1}})

    position_state = PositionState(grid=grid, agents=agents)
    position_state.reset()
    self_observer = PositionCenteredEncodingObserver(agents=agents, grid=grid)
    no_self_observer = PositionCenteredEncodingObserver(
        agents=agents, grid=grid, observe_self=False
    )

    np.testing.assert_array_equal(
        self_observer.get_obs(agents['agent0'])['position_centered_encoding'],
        np.array([
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        no_self_observer.get_obs(agents['agent0'])['position_centered_encoding'],
        np.array([
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        self_observer.get_obs(agents['agent1'])['position_centered_encoding'],
        np.array([
            [-1, -1, -1],
            [-1,  2,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        no_self_observer.get_obs(agents['agent1'])['position_centered_encoding'],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )


def test_absolute_position_observer():
    class PositionObservingAgent(ObservingAgent, GridWorldAgent): pass
    grid = Grid(6, 7, overlapping={1: {5}, 4: {6}, 5: {1}, 6: {4}})
    agents = {
        'agent0': PositionObservingAgent(
            id='agent0',
            encoding=1,
            initial_position=np.array([0, 0])
        ),
        'agent1': PositionObservingAgent(
            id='agent1',
            encoding=2,
            initial_position=np.array([5, 0])
        ),
        'agent2': PositionObservingAgent(
            id='agent2',
            encoding=3,
            initial_position=np.array([0, 6])
        ),
        'agent3': PositionObservingAgent(
            id='agent3',
            encoding=4,
            initial_position=np.array([5, 6])
        ),
        'agent4': PositionObservingAgent(
            id='agent4',
            encoding=5,
            initial_position=np.array([0, 0])
        ),
        'agent5': PositionObservingAgent(
            id='agent5',
            encoding=6,
            initial_position=np.array([5, 6])
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    observer = AbsolutePositionObserver(agents=agents, grid=grid)
    assert observer.key == 'position'
    assert observer.supported_agent_type == ObservingAgent
    assert isinstance(observer, ObserverBaseComponent)
    for agent in agents.values():
        agent.finalize()
        assert agent.observation_space['position'] == Box(
            np.array([0, 0]),
            np.array([5, 6]),
            dtype=int
        )
        np.testing.assert_array_equal(
            agent.null_observation['position'],
            np.array([0, 0])
        )

    position_state.reset()
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['position'],
        np.array([0, 0], dtype=int)
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['position'],
        np.array([5, 0], dtype=int)
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['position'],
        np.array([0, 6], dtype=int)
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent3'])['position'],
        np.array([5, 6], dtype=int)
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent4'])['position'],
        np.array([0, 0], dtype=int)
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent5'])['position'],
        np.array([5, 6], dtype=int)
    )


def test_grid_and_absolute_position_observer_combined():
    grid = Grid(6, 7, overlapping={1: {5}, 4: {6}, 5: {1}, 6: {4}})
    agents = {
        'agent0': GridObservingAgent(
            id='agent0',
            encoding=1,
            initial_position=np.array([2, 2]),
            view_range=2
        ),
        'agent1': GridObservingAgent(
            id='agent1',
            encoding=2,
            initial_position=np.array([3, 4]),
            view_range=2
        ),
        'agent2': GridWorldAgent(
            id='agent2',
            encoding=3,
            initial_position=np.array([0, 0])
        ),
        'agent3': GridWorldAgent(
            id='agent3',
            encoding=4,
            initial_position=np.array([5, 6])
        ),
        'agent4': GridWorldAgent(
            id='agent4',
            encoding=5,
            initial_position=np.array([4, 3])
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    grid_observer = PositionCenteredEncodingObserver(grid=grid, agents=agents)
    position_observer = AbsolutePositionObserver(grid=grid, agents=agents)

    agent = agents['agent0']
    agent.finalize()
    assert agent.observation_space['position'] == Box(
        np.array([0, 0]),
        np.array([5, 6]),
        dtype=int
    )
    assert agent.observation_space['position_centered_encoding'] == Box(-2, 5, (5, 5), int)
    np.testing.assert_array_equal(
        agent.null_observation['position'],
        np.array([0, 0])
    )
    np.testing.assert_array_equal(
        agent.null_observation['position_centered_encoding'],
        -2 * np.ones((5, 5), dtype=int)
    )

    agent = agents['agent1']
    agent.finalize()
    assert agent.observation_space['position'] == Box(
        np.array([0, 0]),
        np.array([5, 6]),
        dtype=int
    )
    assert agent.observation_space['position_centered_encoding'] == Box(-2, 5, (5, 5), int)
    np.testing.assert_array_equal(
        agent.null_observation['position'],
        np.array([0, 0])
    )
    np.testing.assert_array_equal(
        agent.null_observation['position_centered_encoding'],
        -2 * np.ones((5, 5), dtype=int)
    )


    position_state.reset()
    np.testing.assert_array_equal(
        position_observer.get_obs(agents['agent0'])['position'],
        np.array([2, 2], dtype=int)
    )
    np.testing.assert_array_equal(
        grid_observer.get_obs(agents['agent0'])['position_centered_encoding'],
        np.array([
            [3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 5, 0],
        ])
    )
    np.testing.assert_array_equal(
        position_observer.get_obs(agents['agent1'])['position'],
        np.array([3, 4], dtype=int)
    )
    np.testing.assert_array_equal(
        grid_observer.get_obs(agents['agent1'])['position_centered_encoding'],
        np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 5, 0, 0, 0],
            [0, 0, 0, 0, 4],
        ])
    )
