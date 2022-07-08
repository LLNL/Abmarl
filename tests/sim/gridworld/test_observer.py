
from gym.spaces import Box
import numpy as np

from abmarl.sim.gridworld.observer import ObserverBaseComponent, SingleGridObserver, \
    MultiGridObserver
from abmarl.sim.gridworld.agent import GridObservingAgent, GridWorldAgent, MovingAgent
from abmarl.sim.gridworld.state import PositionState
from abmarl.sim.gridworld.grid import Grid


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
    observer = SingleGridObserver(agents=agents, grid=grid)
    assert observer.key == 'grid'
    assert observer.supported_agent_type == GridObservingAgent
    assert isinstance(observer, ObserverBaseComponent)
    assert agents['agent0'].observation_space['grid'] == Box(
        -2, 6, (5, 5), int
    )
    assert agents['agent1'].observation_space['grid'] == Box(
        -2, 6, (3, 3), int
    )
    assert agents['agent2'].observation_space['grid'] == Box(
        -2, 6, (9, 9), int
    )

    agents['agent0'].finalize()
    assert agents['agent0'].null_observation.keys() == set(('grid',))
    np.testing.assert_array_equal(
        agents['agent0'].null_observation['grid'], -2 * np.ones((5, 5), dtype=int)
    )
    agents['agent1'].finalize()
    assert agents['agent1'].null_observation.keys() == set(('grid',))
    np.testing.assert_array_equal(
        agents['agent1'].null_observation['grid'], -2 * np.ones((3, 3), dtype=int)
    )
    agents['agent2'].finalize()
    assert agents['agent2'].null_observation.keys() == set(('grid',))
    np.testing.assert_array_equal(
        agents['agent2'].null_observation['grid'], -2 * np.ones((9, 9), dtype=int)
    )

    position_state.reset()
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'],
        np.array([
            [2, 0, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 6, 1, 0, 0],
            [0, 0, 0, 5, 0],
            [0, 0, 0, 0, 3]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'],
        np.array([
            [-1, -1, -1],
            [-1,  2,  0],
            [-1,  0,  4]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['grid'],
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
    observer = SingleGridObserver(agents=agents, grid=grid)
    assert isinstance(observer, ObserverBaseComponent)
    position_state.reset()

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  4,  0,  0,  0],
            [-2,  6,  1,  0,  0],
            [-2,  0,  0,  5, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'],
        np.array([
            [-1, -1, -1],
            [-1,  2,  0],
            [-1,  0,  4]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['grid'],
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
    grid = Grid(5, 5, overlapping={2: [3], 3: [2], 5: [5]})

    position_state = PositionState(grid=grid, agents=agents)
    observer = MultiGridObserver(agents=agents, grid=grid)
    assert observer.key == 'grid'
    assert observer.supported_agent_type == GridObservingAgent
    assert isinstance(observer, ObserverBaseComponent)
    assert observer.number_of_encodings == 6
    assert agents['agent0'].observation_space['grid'] == Box(
        -2, 9, (5, 5, 6), int
    )
    assert agents['agent1'].observation_space['grid'] == Box(
        -2, 9, (3, 3, 6), int
    )
    assert agents['agent2'].observation_space['grid'] == Box(
        -2, 9, (9, 9, 6), int
    )

    agents['agent0'].finalize()
    assert agents['agent0'].null_observation.keys() == set(('grid',))
    np.testing.assert_array_equal(
        agents['agent0'].null_observation['grid'], -2 * np.ones((5, 5, 6), dtype=int)
    )
    agents['agent1'].finalize()
    assert agents['agent1'].null_observation.keys() == set(('grid',))
    np.testing.assert_array_equal(
        agents['agent1'].null_observation['grid'], -2 * np.ones((3, 3, 6), dtype=int)
    )
    agents['agent2'].finalize()
    assert agents['agent2'].null_observation.keys() == set(('grid',))
    np.testing.assert_array_equal(
        agents['agent2'].null_observation['grid'], -2 * np.ones((9, 9, 6), dtype=int)
    )

    position_state.reset()

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 0],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 1],
        np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 2],
        np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 3],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 4],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 5],
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,0],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,1],
        np.array([
            [-1, -1, -1],
            [-1,  1,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,2],
        np.array([
            [-1, -1, -1],
            [-1,  1,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,3],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,4],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,5],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['grid'][:,:,0],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,1],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,2],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,3],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,4],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,5],
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
    grid = Grid(5, 5, overlapping={2: [3], 3: [2], 5: [5]})

    position_state = PositionState(grid=grid, agents=agents)
    observer = MultiGridObserver(agents=agents, grid=grid)
    assert isinstance(observer, ObserverBaseComponent)
    position_state.reset()

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 0],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  1,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 1],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 2],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 3],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  1,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 4],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  0,  0,  2, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent0'])['grid'][:, :, 5],
        np.array([
            [-2, -2,  0,  0,  0],
            [-2,  0,  0,  0,  0],
            [-2,  1,  0,  0,  0],
            [-2,  0,  0,  0, -2],
            [ 0,  0,  0, -2, -2]
        ])
    )

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,0],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,1],
        np.array([
            [-1, -1, -1],
            [-1,  1,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,2],
        np.array([
            [-1, -1, -1],
            [-1,  1,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,3],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  1]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,4],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        observer.get_obs(agents['agent1'])['grid'][:,:,5],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )

    np.testing.assert_array_equal(
        observer.get_obs(agents['agent2'])['grid'][:,:,0],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,1],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,2],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,3],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,4],
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
        observer.get_obs(agents['agent2'])['grid'][:,:,5],
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
    grid = Grid(5, 5, overlapping={1: [2], 2: [1]})

    position_state = PositionState(grid=grid, agents=agents)
    position_state.reset()
    self_observer = SingleGridObserver(agents=agents, grid=grid)
    no_self_observer = SingleGridObserver(agents=agents, grid=grid, observe_self=False)

    np.testing.assert_array_equal(
        self_observer.get_obs(agents['agent0'])['grid'],
        np.array([
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        no_self_observer.get_obs(agents['agent0'])['grid'],
        np.array([
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    )
    np.testing.assert_array_equal(
        self_observer.get_obs(agents['agent1'])['grid'],
        np.array([
            [-1, -1, -1],
            [-1,  2,  0],
            [-1,  0,  0]
        ])
    )
    np.testing.assert_array_equal(
        no_self_observer.get_obs(agents['agent1'])['grid'],
        np.array([
            [-1, -1, -1],
            [-1,  0,  0],
            [-1,  0,  0]
        ])
    )
