
from gym.spaces import MultiBinary, Dict, Box
import numpy as np

from admiral.envs.components.agent import AgentObservingAgent, PositionAgent, LifeAgent, TeamAgent, SpeedAngleAgent, VelocityAgent
from admiral.envs.components.state import GridPositionState, LifeState, TeamState, ContinuousPositionState, SpeedAngleState, VelocityState
from admiral.envs.components.observer import PositionRestrictedMaskObserver, PositionRestrictedTeamObserver, PositionRestrictedPositionObserver, PositionRestrictedRelativePositionObserver, PositionRestrictedHealthObserver, PositionRestrictedLifeObserver, PositionRestrictedAngleObserver, PositionRestrictedSpeedObserver, PositionRestrictedVelocityObserver

class PositionRestrictedAgent(AgentObservingAgent, PositionAgent, LifeAgent, TeamAgent): pass
class TeamlessAgent(AgentObservingAgent, PositionAgent, LifeAgent): pass
class LifelessAgent(AgentObservingAgent, PositionAgent, TeamAgent): pass
class PositionlessAgent(AgentObservingAgent, LifeAgent, TeamAgent): pass

agents = {
    'agent0':  PositionRestrictedAgent(id='agent0',  agent_view=3, team=0, initial_health=1.0, initial_position=np.array([0, 0])),
    'agent1':  TeamlessAgent(          id='agent1',  agent_view=1,         initial_health=1.0, initial_position=np.array([1, 1])),
    'agent2':  PositionRestrictedAgent(id='agent2',  agent_view=3, team=1, initial_health=1.0, initial_position=np.array([2, 2])),
    'agent3':  LifelessAgent(          id='agent3',  agent_view=4, team=2,                     initial_position=np.array([3, 3])),
    'agent4':  PositionRestrictedAgent(id='agent4',  agent_view=0, team=0, initial_health=1.0, initial_position=np.array([4, 4])),
    'agent5':  PositionRestrictedAgent(id='agent5',  agent_view=3, team=1, initial_health=1.0, initial_position=np.array([5, 5])),
    'agent6':  TeamlessAgent(          id='agent6',  agent_view=2,         initial_health=1.0, initial_position=np.array([6, 6])),
    'agent7':  LifelessAgent(          id='agent7',  agent_view=3, team=2,                     initial_position=np.array([7, 7])),
    'agent8':  PositionRestrictedAgent(id='agent8',  agent_view=5, team=0, initial_health=1.0, initial_position=np.array([8, 8])),
    'agent9':  PositionRestrictedAgent(id='agent9',  agent_view=3, team=1, initial_health=1.0, initial_position=np.array([9, 9])),
    'agent10': PositionlessAgent(      id='agent10', agent_view=3, team=2, initial_health=1.0),
    'agent11': PositionlessAgent(      id='agent11', agent_view=3, team=0, initial_health=1.0),
}

def test_health_restriction():
    position_state = GridPositionState(agents=agents, region=10)
    life_state = LifeState(agents=agents)
    health_observer = PositionRestrictedHealthObserver(agents=agents)

    position_state.reset()
    life_state.reset()

    assert health_observer.get_obs(agents['agent0']) == {'health': {
        'agent0': 1.0,
        'agent1': 1.0,
        'agent2': 1.0,
        'agent4': -1,
        'agent5': -1,
        'agent6': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent1']) == {'health': {
        'agent0': 1.0,
        'agent1': 1.0,
        'agent2': 1.0,
        'agent4': -1,
        'agent5': -1,
        'agent6': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent2']) == {'health': {
        'agent0': 1.0,
        'agent1': 1.0,
        'agent2': 1.0,
        'agent4': 1.0,
        'agent5': 1.0,
        'agent6': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent3']) == {'health': {
        'agent0': 1.0,
        'agent1': 1.0,
        'agent2': 1.0,
        'agent4': 1.0,
        'agent5': 1.0,
        'agent6': 1.0,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent4']) == {'health': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': 1.0,
        'agent5': -1,
        'agent6': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent5']) == {'health': {
        'agent0': -1,
        'agent1': -1,
        'agent2': 1.0,
        'agent4': 1.0,
        'agent5': 1.0,
        'agent6': 1.0,
        'agent8': 1.0,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent6']) == {'health': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': 1.0,
        'agent5': 1.0,
        'agent6': 1.0,
        'agent8': 1.0,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent7']) == {'health': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': 1.0,
        'agent5': 1.0,
        'agent6': 1.0,
        'agent8': 1.0,
        'agent9': 1.0,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent8']) == {'health': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': 1.0,
        'agent5': 1.0,
        'agent6': 1.0,
        'agent8': 1.0,
        'agent9': 1.0,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent9']) == {'health': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': -1,
        'agent5': -1,
        'agent6': 1.0,
        'agent8': 1.0,
        'agent9': 1.0,
        'agent10': -1,
        'agent11': -1,
    }}
    assert health_observer.get_obs(agents['agent10']) == {'health': {
        'agent0': 1.0,
        'agent1': 1.0,
        'agent2': 1.0,
        'agent4': 1.0,
        'agent5': 1.0,
        'agent6': 1.0,
        'agent8': 1.0,
        'agent9': 1.0,
        'agent10': 1.0,
        'agent11': 1.0,
    }}
    assert health_observer.get_obs(agents['agent11']) == {'health': {
        'agent0': 1.0,
        'agent1': 1.0,
        'agent2': 1.0,
        'agent4': 1.0,
        'agent5': 1.0,
        'agent6': 1.0,
        'agent8': 1.0,
        'agent9': 1.0,
        'agent10': 1.0,
        'agent11': 1.0,
    }}

def test_life_restriction():
    position_state = GridPositionState(agents=agents, region=10)
    life_state = LifeState(agents=agents)
    life_observer = PositionRestrictedLifeObserver(agents=agents)

    position_state.reset()
    life_state.reset()
    assert life_observer.get_obs(agents['agent0']) == {'life': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent4': -1,
        'agent5': -1,
        'agent6': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent1']) == {'life': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent4': -1,
        'agent5': -1,
        'agent6': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent2']) == {'life': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent4': True,
        'agent5': True,
        'agent6': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent3']) == {'life': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent4']) == {'life': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': True,
        'agent5': -1,
        'agent6': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent5']) == {'life': {
        'agent0': -1,
        'agent1': -1,
        'agent2': True,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent8': True,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent6']) == {'life': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent8': True,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent7']) == {'life': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent8': True,
        'agent9': True,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent8']) == {'life': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent8': True,
        'agent9': True,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent9']) == {'life': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent4': -1,
        'agent5': -1,
        'agent6': True,
        'agent8': True,
        'agent9': True,
        'agent10': -1,
        'agent11': -1
    }}
    assert life_observer.get_obs(agents['agent10']) == {'life': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent8': True,
        'agent9': True,
        'agent10': True,
        'agent11': True
    }}
    assert life_observer.get_obs(agents['agent11']) == {'life': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent8': True,
        'agent9': True,
        'agent10': True,
        'agent11': True
    }}

def test_team_restriction():
    position_state = GridPositionState(agents=agents, region=10)
    life_state = LifeState(agents=agents)
    team_state = TeamState(agents=agents, number_of_teams=3)
    team_observer = PositionRestrictedTeamObserver(team=team_state, agents=agents)

    position_state.reset()
    life_state.reset()

    assert team_observer.get_obs(agents['agent0']) == {'team': {
        'agent0': 0,
        'agent2': 1,
        'agent3': 2,
        'agent4': -1,
        'agent5': -1,
        'agent7': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent1']) == {'team': {
        'agent0': 0,
        'agent2': 1,
        'agent3': -1,
        'agent4': -1,
        'agent5': -1,
        'agent7': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent2']) == {'team': {
        'agent0': 0,
        'agent2': 1,
        'agent3': 2,
        'agent4': 0,
        'agent5': 1,
        'agent7': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent3']) == {'team': {
        'agent0': 0,
        'agent2': 1,
        'agent3': 2,
        'agent4': 0,
        'agent5': 1,
        'agent7': 2,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent4']) == {'team': {
        'agent0': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': 0,
        'agent5': -1,
        'agent7': -1,
        'agent8': -1,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent5']) == {'team': {
        'agent0': -1,
        'agent2': 1,
        'agent3': 2,
        'agent4': 0,
        'agent5': 1,
        'agent7': 2,
        'agent8': 0,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent6']) == {'team': {
        'agent0': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': 0,
        'agent5': 1,
        'agent7': 2,
        'agent8': 0,
        'agent9': -1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent7']) == {'team': {
        'agent0': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': 0,
        'agent5': 1,
        'agent7': 2,
        'agent8': 0,
        'agent9': 1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent8']) == {'team': {
        'agent0': -1,
        'agent2': -1,
        'agent3': 2,
        'agent4': 0,
        'agent5': 1,
        'agent7': 2,
        'agent8': 0,
        'agent9': 1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent9']) == {'team': {
        'agent0': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': -1,
        'agent5': -1,
        'agent7': 2,
        'agent8': 0,
        'agent9': 1,
        'agent10': -1,
        'agent11': -1
    }}
    assert team_observer.get_obs(agents['agent10']) == {'team': {
        'agent0': 0,
        'agent2': 1,
        'agent3': 2,
        'agent4': 0,
        'agent5': 1,
        'agent7': 2,
        'agent8': 0,
        'agent9': 1,
        'agent10': 2,
        'agent11': 0
    }}
    assert team_observer.get_obs(agents['agent11']) == {'team': {
        'agent0': 0,
        'agent2': 1,
        'agent3': 2,
        'agent4': 0,
        'agent5': 1,
        'agent7': 2,
        'agent8': 0,
        'agent9': 1,
        'agent10': 2,
        'agent11': 0
    }}

def test_position_observer():
    position_state = GridPositionState(agents=agents, region=10)
    life_state = LifeState(agents=agents)
    position_observer = PositionRestrictedPositionObserver(agents=agents, position=position_state)

    position_state.reset()
    life_state.reset()

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent0'], np.array([0, 0]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent1'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent2'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent3'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent4'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent5'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent7'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent8'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent9'], np.array([-1, -1]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent0'], np.array([0, 0]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent1'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent2'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent4'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent5'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent7'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent8'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent9'], np.array([-1, -1]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent0'], np.array([0, 0]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent1'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent2'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent3'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent4'], np.array([4, 4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent5'], np.array([5, 5]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent7'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent8'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent9'], np.array([-1, -1]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent0'], np.array([0, 0]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent1'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent2'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent3'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent4'], np.array([4, 4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent5'], np.array([5, 5]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent6'], np.array([6, 6]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent7'], np.array([7, 7]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent8'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent9'], np.array([-1, -1]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent4'], np.array([4, 4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent5'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent7'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent8'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent9'], np.array([-1, -1]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent2'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent3'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent4'], np.array([4, 4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent5'], np.array([5, 5]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent6'], np.array([6, 6]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent7'], np.array([7, 7]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent8'], np.array([8, 8]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent9'], np.array([-1, -1]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent4'], np.array([4, 4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent5'], np.array([5, 5]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent6'], np.array([6, 6]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent7'], np.array([7, 7]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent8'], np.array([8, 8]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent9'], np.array([-1, -1]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent4'], np.array([4, 4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent5'], np.array([5, 5]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent6'], np.array([6, 6]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent7'], np.array([7, 7]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent8'], np.array([8, 8]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent9'], np.array([9, 9]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent3'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent4'], np.array([4, 4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent5'], np.array([5, 5]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent6'], np.array([6, 6]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent7'], np.array([7, 7]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent8'], np.array([8, 8]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent9'], np.array([9, 9]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent4'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent5'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent6'], np.array([6, 6]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent7'], np.array([7, 7]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent8'], np.array([8, 8]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent9'], np.array([9, 9]))

def test_relative_position_observer():
    position_state = GridPositionState(agents=agents, region=10)
    life_state = LifeState(agents=agents)
    position_observer = PositionRestrictedRelativePositionObserver(agents=agents, position=position_state)

    position_state.reset()
    life_state.reset()

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent1'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent2'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent3'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent4'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent5'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent6'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent7'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent8'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent0'])['position']['agent9'], np.array([-10, -10]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent2'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent3'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent4'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent5'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent6'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent7'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent8'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent1'])['position']['agent9'], np.array([-10, -10]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent0'], np.array([-2, -2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent3'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent4'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent5'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent6'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent7'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent8'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent2'])['position']['agent9'], np.array([-10, -10]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent0'], np.array([-3, -3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent1'], np.array([-2, -2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent4'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent5'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent6'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent7'], np.array([4, 4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent8'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent3'])['position']['agent9'], np.array([-10, -10]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent0'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent1'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent2'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent3'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent5'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent6'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent7'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent8'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent4'])['position']['agent9'], np.array([-10, -10]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent0'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent1'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent2'], np.array([-3, -3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent3'], np.array([-2, -2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent4'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent6'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent7'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent8'], np.array([3, 3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent5'])['position']['agent9'], np.array([-10, -10]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent0'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent1'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent2'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent3'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent4'], np.array([-2, -2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent5'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent7'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent8'], np.array([2, 2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent6'])['position']['agent9'], np.array([-10, -10]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent0'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent1'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent2'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent3'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent4'], np.array([-3, -3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent5'], np.array([-2, -2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent8'], np.array([1, 1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent7'])['position']['agent9'], np.array([2, 2]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent0'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent1'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent2'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent3'], np.array([-5, -5]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent4'], np.array([-4, -4]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent5'], np.array([-3, -3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent6'], np.array([-2, -2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent7'], np.array([-1, -1]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent8'])['position']['agent9'], np.array([1, 1]))

    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent0'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent1'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent2'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent3'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent4'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent5'], np.array([-10, -10]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent6'], np.array([-3, -3]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent7'], np.array([-2, -2]))
    np.testing.assert_array_equal(position_observer.get_obs(agents['agent9'])['position']['agent8'], np.array([-1, -1]))

def test_mask_restriction():
    position_state = GridPositionState(agents=agents, region=10)
    life_state = LifeState(agents=agents)
    mask_observer = PositionRestrictedMaskObserver(agents=agents)

    position_state.reset()
    life_state.reset()

    assert mask_observer.get_obs(agents['agent0']) == {'mask': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent3': True,
        'agent4': False,
        'agent5': False,
        'agent6': False,
        'agent7': False,
        'agent8': False,
        'agent9': False,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent1']) == {'mask': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent3': False,
        'agent4': False,
        'agent5': False,
        'agent6': False,
        'agent7': False,
        'agent8': False,
        'agent9': False,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent2']) == {'mask': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent3': True,
        'agent4': True,
        'agent5': True,
        'agent6': False,
        'agent7': False,
        'agent8': False,
        'agent9': False,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent3']) == {'mask': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent3': True,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent7': True,
        'agent8': False,
        'agent9': False,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent4']) == {'mask': {
        'agent0': False,
        'agent1': False,
        'agent2': False,
        'agent3': False,
        'agent4': True,
        'agent5': False,
        'agent6': False,
        'agent7': False,
        'agent8': False,
        'agent9': False,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent5']) == {'mask': {
        'agent0': False,
        'agent1': False,
        'agent2': True,
        'agent3': True,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent7': True,
        'agent8': True,
        'agent9': False,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent6']) == {'mask': {
        'agent0': False,
        'agent1': False,
        'agent2': False,
        'agent3': False,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent7': True,
        'agent8': True,
        'agent9': False,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent7']) == {'mask': {
        'agent0': False,
        'agent1': False,
        'agent2': False,
        'agent3': False,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent7': True,
        'agent8': True,
        'agent9': True,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent8']) == {'mask': {
        'agent0': False,
        'agent1': False,
        'agent2': False,
        'agent3': True,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent7': True,
        'agent8': True,
        'agent9': True,
        'agent10': False,
        'agent11': False
    }}

    assert mask_observer.get_obs(agents['agent9']) == {'mask': {
        'agent0': False,
        'agent1': False,
        'agent2': False,
        'agent3': False,
        'agent4': False,
        'agent5': False,
        'agent6': True,
        'agent7': True,
        'agent8': True,
        'agent9': True,
        'agent10': False,
        'agent11': False
    }}



class ContinuousPositionRestrictedAgent(AgentObservingAgent, PositionAgent, SpeedAngleAgent): pass

continuous_agents = {
    'agent0': ContinuousPositionRestrictedAgent(id='agent0', agent_view=3, initial_position=np.array([0, 0]), min_speed=0., max_speed=1., max_acceleration=0.5, max_banking_angle=30, max_banking_angle_change=10, initial_speed=0.32, initial_banking_angle=0.0, initial_ground_angle=0),
    'agent1': ContinuousPositionRestrictedAgent(id='agent1', agent_view=1, initial_position=np.array([1, 1]), min_speed=0., max_speed=1., max_acceleration=0.5, max_banking_angle=30, max_banking_angle_change=10, initial_speed=0.24, initial_banking_angle=0.0, initial_ground_angle=90),
    'agent2': ContinuousPositionRestrictedAgent(id='agent2', agent_view=3, initial_position=np.array([2, 2]), min_speed=0., max_speed=1., max_acceleration=0.5, max_banking_angle=30, max_banking_angle_change=10, initial_speed=0.78, initial_banking_angle=0.0, initial_ground_angle=180),
    'agent3': ContinuousPositionRestrictedAgent(id='agent3', agent_view=4, initial_position=np.array([3, 3]), min_speed=0., max_speed=1., max_acceleration=0.5, max_banking_angle=30, max_banking_angle_change=10, initial_speed=0.50, initial_banking_angle=0.0, initial_ground_angle=270),
    'agent4': ContinuousPositionRestrictedAgent(id='agent4', agent_view=0, initial_position=np.array([4, 4]), min_speed=0., max_speed=1., max_acceleration=0.5, max_banking_angle=30, max_banking_angle_change=10, initial_speed=1.00, initial_banking_angle=0.0, initial_ground_angle=24),
    'agent5': ContinuousPositionRestrictedAgent(id='agent5', agent_view=3, initial_position=np.array([5, 5]), min_speed=0., max_speed=1., max_acceleration=0.5, max_banking_angle=30, max_banking_angle_change=10, initial_speed=0.08, initial_banking_angle=0.0, initial_ground_angle=300),
    'agent6': ContinuousPositionRestrictedAgent(id='agent6', agent_view=2, initial_position=np.array([6, 6]), min_speed=0., max_speed=1., max_acceleration=0.5, max_banking_angle=30, max_banking_angle_change=10, initial_speed=0.95, initial_banking_angle=0.0, initial_ground_angle=123),
}

def test_speed_restriction():
    position_state = ContinuousPositionState(agents=continuous_agents, region=10)
    speed_angle_state = SpeedAngleState(agents=continuous_agents)
    speed_observer = PositionRestrictedSpeedObserver(agents=continuous_agents)

    position_state.reset()
    speed_angle_state.reset()

    assert speed_observer.get_obs(agents['agent0']) == {'speed': {
        'agent0': 0.32,
        'agent1': 0.24,
        'agent2': 0.78,
        'agent3': 0.5,
        'agent4': -1,
        'agent5': -1,
        'agent6': -1,
    }}

    assert speed_observer.get_obs(agents['agent1']) == {'speed': {
        'agent0': 0.32,
        'agent1': 0.24,
        'agent2': 0.78,
        'agent3': -1,
        'agent4': -1,
        'agent5': -1,
        'agent6': -1,
    }}

    assert speed_observer.get_obs(agents['agent2']) == {'speed': {
        'agent0': 0.32,
        'agent1': 0.24,
        'agent2': 0.78,
        'agent3': 0.5,
        'agent4': 1.0,
        'agent5': 0.08,
        'agent6': -1,
    }}

    assert speed_observer.get_obs(agents['agent3']) == {'speed': {
        'agent0': 0.32,
        'agent1': 0.24,
        'agent2': 0.78,
        'agent3': 0.5,
        'agent4': 1.0,
        'agent5': 0.08,
        'agent6': 0.95,
    }}

    assert speed_observer.get_obs(agents['agent4']) == {'speed': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': 1.0,
        'agent5': -1,
        'agent6': -1,
    }}

    assert speed_observer.get_obs(agents['agent5']) == {'speed': {
        'agent0': -1,
        'agent1': -1,
        'agent2': 0.78,
        'agent3': 0.5,
        'agent4': 1.0,
        'agent5': 0.08,
        'agent6': 0.95,
    }}

    assert speed_observer.get_obs(agents['agent6']) == {'speed': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': 1.0,
        'agent5': 0.08,
        'agent6': 0.95,
    }}
    
def test_angle_restriction():
    position_state = ContinuousPositionState(agents=continuous_agents, region=10)
    speed_angle_state = SpeedAngleState(agents=continuous_agents)
    angle_observer = PositionRestrictedAngleObserver(agents=continuous_agents)

    position_state.reset()
    speed_angle_state.reset()

    assert angle_observer.get_obs(agents['agent0']) == {'ground_angle': {
        'agent0': 0,
        'agent1': 90,
        'agent2': 180,
        'agent3': 270,
        'agent4': -1,
        'agent5': -1,
        'agent6': -1,
    }}

    assert angle_observer.get_obs(agents['agent1']) == {'ground_angle': {
        'agent0': 0,
        'agent1': 90,
        'agent2': 180,
        'agent3': -1,
        'agent4': -1,
        'agent5': -1,
        'agent6': -1,
    }}

    assert angle_observer.get_obs(agents['agent2']) == {'ground_angle': {
        'agent0': 0,
        'agent1': 90,
        'agent2': 180,
        'agent3': 270,
        'agent4': 24,
        'agent5': 300,
        'agent6': -1,
    }}

    assert angle_observer.get_obs(agents['agent3']) == {'ground_angle': {
        'agent0': 0,
        'agent1': 90,
        'agent2': 180,
        'agent3': 270,
        'agent4': 24,
        'agent5': 300,
        'agent6': 123,
    }}

    assert angle_observer.get_obs(agents['agent4']) == {'ground_angle': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': 24,
        'agent5': -1,
        'agent6': -1,
    }}

    assert angle_observer.get_obs(agents['agent5']) == {'ground_angle': {
        'agent0': -1,
        'agent1': -1,
        'agent2': 180,
        'agent3': 270,
        'agent4': 24,
        'agent5': 300,
        'agent6': 123,
    }}

    assert angle_observer.get_obs(agents['agent6']) == {'ground_angle': {
        'agent0': -1,
        'agent1': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': 24,
        'agent5': 300,
        'agent6': 123,
    }}



class ContinuousVelocityAgent(AgentObservingAgent, PositionAgent, VelocityAgent): pass

def test_velocity_restriction():
    continuous_agents = {
        'agent0': ContinuousVelocityAgent(id='agent0', agent_view=3, initial_position=np.array([0, 0]), max_speed=1., max_acceleration=0.5, initial_velocity=np.array([ 0,  0])),
        'agent1': ContinuousVelocityAgent(id='agent1', agent_view=1, initial_position=np.array([1, 1]), max_speed=1., max_acceleration=0.5, initial_velocity=np.array([ 0,  1])),
        'agent2': ContinuousVelocityAgent(id='agent2', agent_view=3, initial_position=np.array([2, 2]), max_speed=1., max_acceleration=0.5, initial_velocity=np.array([ 1,  0])),
        'agent3': ContinuousVelocityAgent(id='agent3', agent_view=4, initial_position=np.array([3, 3]), max_speed=1., max_acceleration=0.5, initial_velocity=np.array([ 1,  1])),
        'agent4': ContinuousVelocityAgent(id='agent4', agent_view=0, initial_position=np.array([4, 4]), max_speed=1., max_acceleration=0.5, initial_velocity=np.array([ 0, -1])),
        'agent5': ContinuousVelocityAgent(id='agent5', agent_view=3, initial_position=np.array([5, 5]), max_speed=1., max_acceleration=0.5, initial_velocity=np.array([-1,  0])),
        'agent6': ContinuousVelocityAgent(id='agent6', agent_view=2, initial_position=np.array([6, 6]), max_speed=1., max_acceleration=0.5, initial_velocity=np.array([-1, -1])),
    }
    position_state = ContinuousPositionState(agents=continuous_agents, region=10)
    velocity_state = VelocityState(agents=continuous_agents)
    velocity_observer = PositionRestrictedVelocityObserver(agents=continuous_agents)

    position_state.reset()
    velocity_state.reset()

    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent0'])['velocity']['agent0'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent0'])['velocity']['agent1'], np.array([ 0,  1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent0'])['velocity']['agent2'], np.array([ 1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent0'])['velocity']['agent3'], np.array([ 1,  1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent0'])['velocity']['agent4'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent0'])['velocity']['agent5'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent0'])['velocity']['agent6'], np.array([ 0,  0]))

    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent1'])['velocity']['agent0'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent1'])['velocity']['agent1'], np.array([ 0,  1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent1'])['velocity']['agent2'], np.array([ 1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent1'])['velocity']['agent3'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent1'])['velocity']['agent4'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent1'])['velocity']['agent5'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent1'])['velocity']['agent6'], np.array([ 0,  0]))

    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent2'])['velocity']['agent0'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent2'])['velocity']['agent1'], np.array([ 0,  1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent2'])['velocity']['agent2'], np.array([ 1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent2'])['velocity']['agent3'], np.array([ 1,  1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent2'])['velocity']['agent4'], np.array([ 0, -1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent2'])['velocity']['agent5'], np.array([-1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent2'])['velocity']['agent6'], np.array([ 0,  0]))

    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent3'])['velocity']['agent0'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent3'])['velocity']['agent1'], np.array([ 0,  1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent3'])['velocity']['agent2'], np.array([ 1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent3'])['velocity']['agent3'], np.array([ 1,  1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent3'])['velocity']['agent4'], np.array([ 0, -1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent3'])['velocity']['agent5'], np.array([-1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent3'])['velocity']['agent6'], np.array([-1, -1]))

    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent4'])['velocity']['agent0'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent4'])['velocity']['agent1'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent4'])['velocity']['agent2'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent4'])['velocity']['agent3'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent4'])['velocity']['agent4'], np.array([ 0, -1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent4'])['velocity']['agent5'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent4'])['velocity']['agent6'], np.array([ 0,  0]))

    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent5'])['velocity']['agent0'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent5'])['velocity']['agent1'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent5'])['velocity']['agent2'], np.array([ 1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent5'])['velocity']['agent3'], np.array([ 1,  1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent5'])['velocity']['agent4'], np.array([ 0, -1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent5'])['velocity']['agent5'], np.array([-1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent5'])['velocity']['agent6'], np.array([-1, -1]))

    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent6'])['velocity']['agent0'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent6'])['velocity']['agent1'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent6'])['velocity']['agent2'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent6'])['velocity']['agent3'], np.array([ 0,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent6'])['velocity']['agent4'], np.array([ 0, -1]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent6'])['velocity']['agent5'], np.array([-1,  0]))
    np.testing.assert_array_equal(velocity_observer.get_obs(agents['agent6'])['velocity']['agent6'], np.array([-1, -1]))
