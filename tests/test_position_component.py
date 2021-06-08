import numpy as np

from abmarl.sim.components.state import GridPositionState, LifeState
from abmarl.sim.components.observer import GridPositionBasedObserver, \
    GridPositionTeamBasedObserver, RelativePositionObserver
from abmarl.sim.components.agent import PositionObservingAgent, AgentObservingAgent, \
    ComponentAgent


class PositionTestAgent(PositionObservingAgent, AgentObservingAgent): pass
class PositionTeamTestAgent(PositionObservingAgent, AgentObservingAgent): pass
class PositionTeamNoViewTestAgent(ComponentAgent): pass


def test_grid_position_observer():
    agents = {
        'agent0': PositionTestAgent(id='agent0', initial_position=np.array([0, 0]), agent_view=1),
        'agent1': PositionTestAgent(id='agent1', initial_position=np.array([2, 2]), agent_view=2),
        'agent2': PositionTestAgent(id='agent2', initial_position=np.array([3, 2]), agent_view=3),
        'agent3': PositionTestAgent(id='agent3', initial_position=np.array([1, 4]), agent_view=4),
        'agent4': ComponentAgent(id='agent4', initial_position=np.array([1, 4])),
    }

    state = GridPositionState(agents=agents, region=5)
    life = LifeState(agents=agents)
    observer = GridPositionBasedObserver(position_state=state, agents=agents)
    state.reset()
    life.reset()

    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])['position'], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent1'])['position'], np.array([
        [1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])['position'], np.array([
        [-1.,  1.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  1., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])['position'], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 1.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  1., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    ]))
    assert observer.get_obs(agents['agent4']) == {}


def test_grid_team_position_observer():
    agents = {
        'agent0': PositionTeamTestAgent(
            id='agent0', team=1, initial_position=np.array([0, 0]), agent_view=1
        ),
        'agent1': PositionTeamNoViewTestAgent(
            id='agent1', team=1, initial_position=np.array([0, 0])
        ),
        'agent2': PositionTeamTestAgent(
            id='agent2', team=1, initial_position=np.array([2, 2]), agent_view=2
        ),
        'agent3': PositionTeamTestAgent(
            id='agent3', team=2, initial_position=np.array([3, 2]), agent_view=3
        ),
        'agent4': PositionTeamTestAgent(
            id='agent4', team=2, initial_position=np.array([1, 4]), agent_view=4
        ),
        'agent5': PositionTeamNoViewTestAgent(
            id='agent5', team=2, initial_position=np.array([1, 4])
        ),
        'agent6': PositionTeamNoViewTestAgent(
            id='agent6', team=2, initial_position=np.array([1, 4])
        ),
        'agent7': PositionTeamTestAgent(
            id='agent7', team=3, initial_position=np.array([1, 4]), agent_view=2
        ),
    }
    for agent in agents.values():
        agent.position = agent.initial_position

    state = GridPositionState(agents=agents, region=5)
    life = LifeState(agents=agents)
    observer = GridPositionTeamBasedObserver(
        position_state=state, number_of_teams=3, agents=agents
    )
    state.reset()
    life.reset()

    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])['position'][:,:,1], np.array([
        [-1., -1., -1.],
        [-1.,  1.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])['position'][:,:,2], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])['position'][:,:,3], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])['position'][:,:,1], np.array([
        [2., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])['position'][:,:,2], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 3.],
        [0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])['position'][:,:,3], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])['position'][:,:,1], np.array([
        [-1.,  2.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])['position'][:,:,2], np.array([
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  3., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])['position'][:,:,3], np.array([
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 2.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])['position'][:,:,2], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  2., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])['position'][:,:,3], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  1., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 1.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])['position'][:,:,2], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  3., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 1.,  0.,  0., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])['position'][:,:,3], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
    ]))


def test_relative_position_observer():
    agents = {
        'agent0': PositionTestAgent(id='agent0', initial_position=np.array([0, 0]), agent_view=1),
        'agent1': PositionTestAgent(id='agent1', initial_position=np.array([2, 2]), agent_view=2),
        'agent2': PositionTestAgent(id='agent2', initial_position=np.array([3, 2]), agent_view=3),
        'agent3': PositionTestAgent(id='agent3', initial_position=np.array([1, 4]), agent_view=4),
        'agent4': ComponentAgent(id='agent4', initial_position=np.array([1, 4])),
    }

    state = GridPositionState(agents=agents, region=5)
    observer = RelativePositionObserver(position_state=state, agents=agents)
    state.reset()

    assert observer.get_obs(agents['agent0'])['relative_position']['agent1'][0] == 2
    assert observer.get_obs(agents['agent0'])['relative_position']['agent1'][1] == 2
    assert observer.get_obs(agents['agent0'])['relative_position']['agent2'][0] == 3
    assert observer.get_obs(agents['agent0'])['relative_position']['agent2'][1] == 2
    assert observer.get_obs(agents['agent0'])['relative_position']['agent3'][0] == 1
    assert observer.get_obs(agents['agent0'])['relative_position']['agent3'][1] == 4
    assert observer.get_obs(agents['agent0'])['relative_position']['agent4'][0] == 1
    assert observer.get_obs(agents['agent0'])['relative_position']['agent4'][1] == 4

    assert observer.get_obs(agents['agent1'])['relative_position']['agent0'][0] == -2
    assert observer.get_obs(agents['agent1'])['relative_position']['agent0'][1] == -2
    assert observer.get_obs(agents['agent1'])['relative_position']['agent2'][0] == 1
    assert observer.get_obs(agents['agent1'])['relative_position']['agent2'][1] == 0
    assert observer.get_obs(agents['agent1'])['relative_position']['agent3'][0] == -1
    assert observer.get_obs(agents['agent1'])['relative_position']['agent3'][1] == 2
    assert observer.get_obs(agents['agent1'])['relative_position']['agent4'][0] == -1
    assert observer.get_obs(agents['agent1'])['relative_position']['agent4'][1] == 2

    assert observer.get_obs(agents['agent2'])['relative_position']['agent0'][0] == -3
    assert observer.get_obs(agents['agent2'])['relative_position']['agent0'][1] == -2
    assert observer.get_obs(agents['agent2'])['relative_position']['agent1'][0] == -1
    assert observer.get_obs(agents['agent2'])['relative_position']['agent1'][1] == 0
    assert observer.get_obs(agents['agent2'])['relative_position']['agent3'][0] == -2
    assert observer.get_obs(agents['agent2'])['relative_position']['agent3'][1] == 2
    assert observer.get_obs(agents['agent2'])['relative_position']['agent4'][0] == -2
    assert observer.get_obs(agents['agent2'])['relative_position']['agent4'][1] == 2

    assert observer.get_obs(agents['agent3'])['relative_position']['agent0'][0] == -1
    assert observer.get_obs(agents['agent3'])['relative_position']['agent0'][1] == -4
    assert observer.get_obs(agents['agent3'])['relative_position']['agent1'][0] == 1
    assert observer.get_obs(agents['agent3'])['relative_position']['agent1'][1] == -2
    assert observer.get_obs(agents['agent3'])['relative_position']['agent2'][0] == 2
    assert observer.get_obs(agents['agent3'])['relative_position']['agent2'][1] == -2
    assert observer.get_obs(agents['agent3'])['relative_position']['agent4'][0] == 0
    assert observer.get_obs(agents['agent3'])['relative_position']['agent4'][1] == 0
