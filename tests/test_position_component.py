
from gym.spaces import Box

import numpy as np

from admiral.envs.components.position import GridPositionAgent, TeamAgent, ObservingAgent
from admiral.envs.components.position import GridPositionComponent, GridPositionTeamsComponent

class PositionTestAgent(GridPositionAgent, ObservingAgent): pass
class PositionTeamTestAgent(GridPositionAgent, ObservingAgent, TeamAgent): pass
class PositionTeamNoViewTestAgent(GridPositionAgent, TeamAgent): pass

def test_grid_position_component():
    agents = {
        'agent0': PositionTestAgent(id='agent0', starting_position=np.array([0, 0]), view=1),
        'agent1': PositionTestAgent(id='agent1', starting_position=np.array([2, 2]), view=2),
        'agent2': PositionTestAgent(id='agent2', starting_position=np.array([3, 2]), view=3),
        'agent3': PositionTestAgent(id='agent3', starting_position=np.array([1, 4]), view=4),
        'agent4': GridPositionAgent(id='agent4', starting_position=np.array([1, 4])),
    }
    
    component = GridPositionComponent(agents=agents, region=5)
    for agent in agents.values():
        agent.position = agent.starting_position
        if isinstance(agent, ObservingAgent):
            assert agent.observation_space['agents'] == Box(-1, 1, (agent.view*2+1, agent.view*2+1), np.int)
    np.testing.assert_array_equal(component.get_obs('agent0'), np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent1'), np.array([
        [1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent2'), np.array([
        [-1.,  1.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  1., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent3'), np.array([
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


def test_grid_team_position_component():
    agents = {
        'agent0': PositionTeamTestAgent(id='agent0', team=0, starting_position=np.array([0, 0]), view=1),
        'agent1': PositionTeamNoViewTestAgent(id='agent1', team=0, starting_position=np.array([0, 0])),
        'agent2': PositionTeamTestAgent(id='agent2', team=0, starting_position=np.array([2, 2]), view=2),
        'agent3': PositionTeamTestAgent(id='agent3', team=1, starting_position=np.array([3, 2]), view=3),
        'agent4': PositionTeamTestAgent(id='agent4', team=1, starting_position=np.array([1, 4]), view=4),
        'agent5': PositionTeamNoViewTestAgent(id='agent5', team=1, starting_position=np.array([1, 4])),
        'agent6': PositionTeamNoViewTestAgent(id='agent6', team=1, starting_position=np.array([1, 4])),
        'agent7': PositionTeamTestAgent(id='agent7', team=2, starting_position=np.array([1, 4]), view=2),
    }
    for agent in agents.values():
        agent.position = agent.starting_position
    
    component = GridPositionTeamsComponent(agents=agents, region=5, number_of_teams=3)

    np.testing.assert_array_equal(component.get_obs('agent0')[:,:,0], np.array([
        [-1., -1., -1.],
        [-1.,  1.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent0')[:,:,1], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent0')[:,:,2], np.array([
        [-1., -1., -1.],
        [-1.,  0.,  0.],
        [-1.,  0.,  0.],
    ]))

    np.testing.assert_array_equal(component.get_obs('agent2')[:,:,0], np.array([
        [2., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent2')[:,:,1], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 3.],
        [0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent2')[:,:,2], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))

    np.testing.assert_array_equal(component.get_obs('agent3')[:,:,0], np.array([
        [-1.,  2.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent3')[:,:,1], np.array([
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  3., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent3')[:,:,2], np.array([
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1.],
    ]))

    np.testing.assert_array_equal(component.get_obs('agent4')[:,:,0], np.array([
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
    np.testing.assert_array_equal(component.get_obs('agent4')[:,:,1], np.array([
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
    np.testing.assert_array_equal(component.get_obs('agent4')[:,:,2], np.array([
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

    np.testing.assert_array_equal(component.get_obs('agent7')[:,:,0], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 1.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent7')[:,:,1], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  3., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 1.,  0.,  0., -1., -1.],
    ]))
    np.testing.assert_array_equal(component.get_obs('agent7')[:,:,2], np.array([
        [-1., -1., -1., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0., -1., -1.],
    ]))
