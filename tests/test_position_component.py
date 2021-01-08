
from gym.spaces import Box

import numpy as np

from admiral.envs.components.agent import TeamAgent, PositionAgent, AgentObservingAgent, LifeAgent
from admiral.envs.components.state import TeamState, PositionState, LifeState
from admiral.envs.components.observer import PositionObserver, GridPositionBasedObserver, GridPositionTeamBasedObserver, RelativePositionObserver

class PositionTestAgent(PositionAgent, AgentObservingAgent, LifeAgent): pass
class PositionTeamTestAgent(PositionAgent, AgentObservingAgent, TeamAgent, LifeAgent): pass
class PositionTeamNoViewTestAgent(PositionAgent, TeamAgent, LifeAgent): pass
class PositionLifeAgent(PositionAgent, LifeAgent): pass

def test_position_observer():
    pass # TODO: Implement position based observer of the agents

def test_grid_position_observer():
    agents = {
        'agent0': PositionTestAgent(id='agent0', starting_position=np.array([0, 0]), agent_view=1),
        'agent1': PositionTestAgent(id='agent1', starting_position=np.array([2, 2]), agent_view=2),
        'agent2': PositionTestAgent(id='agent2', starting_position=np.array([3, 2]), agent_view=3),
        'agent3': PositionTestAgent(id='agent3', starting_position=np.array([1, 4]), agent_view=4),
        'agent4': PositionLifeAgent(id='agent4', starting_position=np.array([1, 4])),
    }
    
    state = PositionState(agents=agents, region=5)
    life = LifeState(agents=agents)
    observer = GridPositionBasedObserver(position=state, agents=agents)
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
        'agent0': PositionTeamTestAgent      (id='agent0', team=0, starting_position=np.array([0, 0]), agent_view=1),
        'agent1': PositionTeamNoViewTestAgent(id='agent1', team=0, starting_position=np.array([0, 0])),
        'agent2': PositionTeamTestAgent      (id='agent2', team=0, starting_position=np.array([2, 2]), agent_view=2),
        'agent3': PositionTeamTestAgent      (id='agent3', team=1, starting_position=np.array([3, 2]), agent_view=3),
        'agent4': PositionTeamTestAgent      (id='agent4', team=1, starting_position=np.array([1, 4]), agent_view=4),
        'agent5': PositionTeamNoViewTestAgent(id='agent5', team=1, starting_position=np.array([1, 4])),
        'agent6': PositionTeamNoViewTestAgent(id='agent6', team=1, starting_position=np.array([1, 4])),
        'agent7': PositionTeamTestAgent      (id='agent7', team=2, starting_position=np.array([1, 4]), agent_view=2),
    }
    for agent in agents.values():
        agent.position = agent.starting_position
    
    state = PositionState(agents=agents, region=5)
    life = LifeState(agents=agents)
    team = TeamState(agents=agents, number_of_teams=3)
    observer = GridPositionTeamBasedObserver(position=state, team_state=team, agents=agents)
    state.reset()
    life.reset()

    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])['position'][:,:,0], np.array([
        [-1., -1., -1.],
        [-1.,  1.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])['position'][:,:,1], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent0'])['position'][:,:,2], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])['position'][:,:,0], np.array([
        [2., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])['position'][:,:,1], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 3.],
        [0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent2'])['position'][:,:,2], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])['position'][:,:,0], np.array([
        [-1.,  2.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])['position'][:,:,1], np.array([
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  3., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent3'])['position'][:,:,2], np.array([
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))

    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])['position'][:,:,0], np.array([
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
    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])['position'][:,:,1], np.array([
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
    np.testing.assert_array_equal(observer.get_obs(agents['agent4'])['position'][:,:,2], np.array([
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

    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 1.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  3., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 1.,  0.,  0., -1., -1.],
    ]))
    np.testing.assert_array_equal(observer.get_obs(agents['agent7'])['position'][:,:,2], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
    ]))

def test_relative_position_observer():
    agents = {
        'agent0': PositionTestAgent(id='agent0', starting_position=np.array([0, 0]), agent_view=1),
        'agent1': PositionTestAgent(id='agent1', starting_position=np.array([2, 2]), agent_view=2),
        'agent2': PositionTestAgent(id='agent2', starting_position=np.array([3, 2]), agent_view=3),
        'agent3': PositionTestAgent(id='agent3', starting_position=np.array([1, 4]), agent_view=4),
        'agent4': PositionAgent(id='agent4', starting_position=np.array([1, 4])),
    }
    
    state = PositionState(agents=agents, region=5)
    observer = RelativePositionObserver(position=state, agents=agents)
    state.reset()

    assert observer.get_obs(agents['agent0'])['position']['agent1'][0] == 2
    assert observer.get_obs(agents['agent0'])['position']['agent1'][1] == 2
    assert observer.get_obs(agents['agent0'])['position']['agent2'][0] == 3
    assert observer.get_obs(agents['agent0'])['position']['agent2'][1] == 2
    assert observer.get_obs(agents['agent0'])['position']['agent3'][0] == 1
    assert observer.get_obs(agents['agent0'])['position']['agent3'][1] == 4
    assert observer.get_obs(agents['agent0'])['position']['agent4'][0] == 1
    assert observer.get_obs(agents['agent0'])['position']['agent4'][1] == 4

    assert observer.get_obs(agents['agent1'])['position']['agent0'][0] == -2
    assert observer.get_obs(agents['agent1'])['position']['agent0'][1] == -2
    assert observer.get_obs(agents['agent1'])['position']['agent2'][0] == 1
    assert observer.get_obs(agents['agent1'])['position']['agent2'][1] == 0
    assert observer.get_obs(agents['agent1'])['position']['agent3'][0] == -1
    assert observer.get_obs(agents['agent1'])['position']['agent3'][1] == 2
    assert observer.get_obs(agents['agent1'])['position']['agent4'][0] == -1
    assert observer.get_obs(agents['agent1'])['position']['agent4'][1] == 2

    assert observer.get_obs(agents['agent2'])['position']['agent0'][0] == -3
    assert observer.get_obs(agents['agent2'])['position']['agent0'][1] == -2
    assert observer.get_obs(agents['agent2'])['position']['agent1'][0] == -1
    assert observer.get_obs(agents['agent2'])['position']['agent1'][1] == 0
    assert observer.get_obs(agents['agent2'])['position']['agent3'][0] == -2
    assert observer.get_obs(agents['agent2'])['position']['agent3'][1] == 2
    assert observer.get_obs(agents['agent2'])['position']['agent4'][0] == -2
    assert observer.get_obs(agents['agent2'])['position']['agent4'][1] == 2

    assert observer.get_obs(agents['agent3'])['position']['agent0'][0] == -1
    assert observer.get_obs(agents['agent3'])['position']['agent0'][1] == -4
    assert observer.get_obs(agents['agent3'])['position']['agent1'][0] == 1
    assert observer.get_obs(agents['agent3'])['position']['agent1'][1] == -2
    assert observer.get_obs(agents['agent3'])['position']['agent2'][0] == 2
    assert observer.get_obs(agents['agent3'])['position']['agent2'][1] == -2
    assert observer.get_obs(agents['agent3'])['position']['agent4'][0] == 0
    assert observer.get_obs(agents['agent3'])['position']['agent4'][1] == 0
