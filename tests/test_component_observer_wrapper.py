
import numpy as np

from admiral.envs.components.agent import PositionAgent, LifeAgent, TeamAgent, SpeedAngleAgent, VelocityAgent
from admiral.envs.components.state import GridPositionState, LifeState, TeamState, ContinuousPositionState, SpeedAngleState, VelocityState
from admiral.envs.components.observer import HealthObserver, LifeObserver, MaskObserver, \
    PositionObserver, RelativePositionObserver, SpeedObserver, AngleObserver, VelocityObserver, \
    TeamObserver
from admiral.envs.components.wrappers.observer_wrapper import PositionRestrictedObservationWrapper

from admiral.envs.components.agent import AgentObservingAgent, VelocityObservingAgent, PositionObservingAgent, SpeedAngleObservingAgent, TeamObservingAgent, LifeObservingAgent, HealthObservingAgent
class AllObservingAgent(AgentObservingAgent, VelocityObservingAgent, PositionObservingAgent, SpeedAngleObservingAgent, TeamObservingAgent, LifeObservingAgent, HealthObservingAgent): pass

class AllAgent(           AllObservingAgent, PositionAgent, LifeAgent, TeamAgent, SpeedAngleAgent, VelocityAgent): pass
class PositionlessAgent(  AllObservingAgent,                LifeAgent, TeamAgent, SpeedAngleAgent, VelocityAgent): pass
class LifelessAgent(      AllObservingAgent, PositionAgent,            TeamAgent, SpeedAngleAgent, VelocityAgent): pass
class TeamlessAgent(      AllObservingAgent, PositionAgent, LifeAgent,            SpeedAngleAgent, VelocityAgent): pass
class SpeedAnglelessAgent(AllObservingAgent, PositionAgent, LifeAgent, TeamAgent,                  VelocityAgent): pass
class VelocitylessAgent(  AllObservingAgent, PositionAgent, LifeAgent, TeamAgent, SpeedAngleAgent               ): pass

agents = {
    'agent0': AllAgent(           id='agent0', agent_view=2, initial_position=np.array([2, 2]), initial_health=0.67, team=0, max_speed=1, initial_speed=0.30, initial_banking_angle=7,  initial_ground_angle=123, initial_velocity=np.array([-0.3, 0.8])),
    'agent1': AllAgent(           id='agent1', agent_view=1, initial_position=np.array([4, 4]), initial_health=0.54, team=1, max_speed=2, initial_speed=0.00, initial_banking_angle=0,  initial_ground_angle=126, initial_velocity=np.array([-0.2, 0.7])),
    'agent2': AllAgent(           id='agent2', agent_view=1, initial_position=np.array([4, 3]), initial_health=0.36, team=1, max_speed=1, initial_speed=0.12, initial_banking_angle=30, initial_ground_angle=180, initial_velocity=np.array([-0.1, 0.6])),
    'agent3': AllAgent(           id='agent3', agent_view=1, initial_position=np.array([4, 2]), initial_health=0.24, team=1, max_speed=4, initial_speed=0.05, initial_banking_angle=13, initial_ground_angle=46,  initial_velocity=np.array([0.0,  0.5])),
    'agent4': LifelessAgent(      id='agent4', agent_view=1, initial_position=np.array([4, 1]),                      team=2, max_speed=3, initial_speed=0.17, initial_banking_angle=15, initial_ground_angle=212, initial_velocity=np.array([0.1,  0.4])),
    'agent5': TeamlessAgent(      id='agent5', agent_view=1, initial_position=np.array([4, 0]), initial_health=0.89,         max_speed=2, initial_speed=0.21, initial_banking_angle=23, initial_ground_angle=276, initial_velocity=np.array([0.2,  0.3])),
    'agent6': SpeedAnglelessAgent(id='agent6', agent_view=0, initial_position=np.array([1, 1]), initial_health=0.53, team=0, max_speed=1,                                                                         initial_velocity=np.array([0.3,  0.2])),
    'agent7': VelocitylessAgent(  id='agent7', agent_view=5, initial_position=np.array([0, 4]), initial_health=0.50, team=1, max_speed=1, initial_speed=0.36, initial_banking_angle=24, initial_ground_angle=0                                          ),
    'agent8': PositionlessAgent(  id='agent8', agent_view=1,                                    initial_health=0.26, team=0, max_speed=2, initial_speed=0.06, initial_banking_angle=16, initial_ground_angle=5,   initial_velocity=np.array([0.5,  0.0])),
    'agent9': PositionlessAgent(  id='agent9', agent_view=1,                                    initial_health=0.08, team=2, max_speed=0, initial_speed=0.24, initial_banking_angle=30, initial_ground_angle=246, initial_velocity=np.array([0.6, -0.1])),
}

def linear_drop_off(distance, view):
    return 1. - 1. / (view+1) * distance

def test_health_restriction():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=5)
    life_state = LifeState(agents=agents)
    health_observer = PositionRestrictedObservationWrapper(HealthObserver(agents=agents), obs_filter=linear_drop_off, agents=agents)

    position_state.reset()
    life_state.reset()

    assert health_observer.get_obs(agents['agent0']) == {'health': {
        'agent0': 0.67,
        'agent1': 0.54,
        'agent2': -1,
        'agent3': 0.24,
        'agent5': 0.89,
        'agent6': -1,
        'agent7': -1, 
        'agent8': 0.26,
        'agent9': 0.08,
    }}
    assert health_observer.get_obs(agents['agent1']) == {'health': {
        'agent0': -1,
        'agent1': 0.54,
        'agent2': -1,
        'agent3': -1,
        'agent5': -1,
        'agent6': -1,
        'agent7': -1,
        'agent8': 0.26,
        'agent9': 0.08,
    }}
    assert health_observer.get_obs(agents['agent9']) == {'health': {
        'agent0': 0.67,
        'agent1': 0.54,
        'agent2': 0.36,
        'agent3': 0.24,
        'agent5': 0.89,
        'agent6': 0.53,
        'agent7': 0.5,
        'agent8': 0.26,
        'agent9': 0.08,
    }}

def test_life_restriction():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=10)
    life_state = LifeState(agents=agents)
    life_observer = PositionRestrictedObservationWrapper(LifeObserver(agents=agents), obs_filter=linear_drop_off, agents=agents)

    position_state.reset()
    life_state.reset()

    assert life_observer.get_obs(agents['agent0']) == {'life': {
        'agent0': True,
        'agent1': True,
        'agent2': -1,
        'agent3': True,
        'agent5': True,
        'agent6': -1,
        'agent7': -1,
        'agent8': True,
        'agent9': True,
    }}
    assert life_observer.get_obs(agents['agent1']) == {'life': {
        'agent0': -1,
        'agent1': True,
        'agent2': -1,
        'agent3': -1,
        'agent5': -1,
        'agent6': -1,
        'agent7': -1,
        'agent8': True,
        'agent9': True,
    }}
    assert life_observer.get_obs(agents['agent9']) == {'life': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent3': True,
        'agent5': True,
        'agent6': True,
        'agent7': True,
        'agent8': True,
        'agent9': True,
    }}

def test_team_restriction():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=10)
    team_state = TeamState(agents=agents, number_of_teams=3)
    team_observer = PositionRestrictedObservationWrapper(TeamObserver(team=team_state, agents=agents), obs_filter=linear_drop_off, agents=agents)

    position_state.reset()

    assert team_observer.get_obs(agents['agent0']) == {'team': {
        'agent0': 0,
        'agent1': 1,
        'agent2': -1,
        'agent3': 1,
        'agent4': -1,
        'agent6': -1,
        'agent7': -1,
        'agent8': 0,
        'agent9': 2,
    }}
    assert team_observer.get_obs(agents['agent1']) == {'team': {
        'agent0': -1,
        'agent1': 1,
        'agent2': -1,
        'agent3': -1,
        'agent4': -1,
        'agent6': -1,
        'agent7': -1,
        'agent8': 0,
        'agent9': 2,
    }}
    assert team_observer.get_obs(agents['agent9']) == {'team': {
        'agent0': 0,
        'agent1': 1,
        'agent2': 1,
        'agent3': 1,
        'agent4': 2,
        'agent6': 0,
        'agent7': 1,
        'agent8': 0,
        'agent9': 2,
    }}


def test_position_observer():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=10)
    position_observer = PositionRestrictedObservationWrapper(PositionObserver(position=position_state, agents=agents), obs_filter=linear_drop_off, agents=agents)

    position_state.reset()

    obs = position_observer.get_obs(agents['agent0'])
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([2, 2]))
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([4, 4]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([4, 2]))
    np.testing.assert_array_equal(obs['position']['agent4'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent5'], np.array([4, 0]))
    np.testing.assert_array_equal(obs['position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent7'], np.array([-1, -1]))
    assert 'agent8' not in obs['position']
    assert 'agent9' not in obs['position']


    obs = position_observer.get_obs(agents['agent1'])
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([4, 4]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent4'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent5'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent7'], np.array([-1, -1]))
    assert 'agent8' not in obs['position']
    assert 'agent9' not in obs['position']


    obs = position_observer.get_obs(agents['agent9'])
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([2, 2]))
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([4, 4]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([4, 3]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([4, 2]))
    np.testing.assert_array_equal(obs['position']['agent4'], np.array([4, 1]))
    np.testing.assert_array_equal(obs['position']['agent5'], np.array([4, 0]))
    np.testing.assert_array_equal(obs['position']['agent6'], np.array([1, 1]))
    np.testing.assert_array_equal(obs['position']['agent7'], np.array([0, 4]))
    assert 'agent8' not in obs['position']
    assert 'agent9' not in obs['position']


def test_relative_position_observer():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=10)
    position_observer = PositionRestrictedObservationWrapper(RelativePositionObserver(position=position_state, agents=agents), obs_filter=linear_drop_off, agents=agents)

    position_state.reset()
    
    obs = position_observer.get_obs(agents['agent0'])
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([2, 2]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([2, 0]))
    np.testing.assert_array_equal(obs['position']['agent4'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent5'], np.array([2, -2]))
    np.testing.assert_array_equal(obs['position']['agent6'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent7'], np.array([-10, -10]))
    assert 'agent0' not in obs['position']
    assert 'agent8' not in obs['position']
    assert 'agent9' not in obs['position']

    
    obs = position_observer.get_obs(agents['agent1'])
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent4'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent5'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent6'], np.array([-10, -10]))
    np.testing.assert_array_equal(obs['position']['agent7'], np.array([-10, -10]))
    assert 'agent1' not in obs['position']
    assert 'agent8' not in obs['position']
    assert 'agent9' not in obs['position']


def test_mask_restriction():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=10)
    mask_observer = PositionRestrictedObservationWrapper(MaskObserver(agents=agents), obs_filter=linear_drop_off, agents=agents)

    position_state.reset()

    assert mask_observer.get_obs(agents['agent0']) == {'mask': {
        'agent0': True,
        'agent1': True,
        'agent2': False,
        'agent3': True,
        'agent4': False,
        'agent5': True,
        'agent6': False,
        'agent7': False,
        'agent8': True,
        'agent9': True,
    }}

    assert mask_observer.get_obs(agents['agent1']) == {'mask': {
        'agent0': False,
        'agent1': True,
        'agent2': False,
        'agent3': False,
        'agent4': False,
        'agent5': False,
        'agent6': False,
        'agent7': False,
        'agent8': True,
        'agent9': True,
    }}

    assert mask_observer.get_obs(agents['agent9']) == {'mask': {
        'agent0': True,
        'agent1': True,
        'agent2': True,
        'agent3': True,
        'agent4': True,
        'agent5': True,
        'agent6': True,
        'agent7': True,
        'agent8': True,
        'agent9': True,
    }}


def test_speed_restriction():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=5)
    speed_state = SpeedAngleState(agents=agents)
    speed_observer = PositionRestrictedObservationWrapper(SpeedObserver(agents=agents), obs_filter=linear_drop_off, agents=agents)

    position_state.reset()
    speed_state.reset()

    assert speed_observer.get_obs(agents['agent0']) == {'speed': {
        'agent0': 0.3,
        'agent1': 0.0,
        'agent2': -1,
        'agent3': 0.05,
        'agent4': -1,
        'agent5': 0.21,
        'agent7': -1,
        'agent8': 0.06,
        'agent9': 0.24,
    }}

    assert speed_observer.get_obs(agents['agent1']) == {'speed': {
        'agent0': -1,
        'agent1': 0.0,
        'agent2': -1,
        'agent3': -1,
        'agent4': -1,
        'agent5': -1,
        'agent7': -1,
        'agent8': 0.06,
        'agent9': 0.24,
    }}

    assert speed_observer.get_obs(agents['agent9']) == {'speed': {
        'agent0': 0.3,
        'agent1': 0.0,
        'agent2': 0.12,
        'agent3': 0.05,
        'agent4': 0.17,
        'agent5': 0.21,
        'agent7': 0.36,
        'agent8': 0.06,
        'agent9': 0.24,
    }}


def test_speed_restriction():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=5)
    angle_state = SpeedAngleState(agents=agents)
    angle_observer = PositionRestrictedObservationWrapper(AngleObserver(agents=agents), obs_filter=linear_drop_off, agents=agents)

    position_state.reset()
    angle_state.reset()

    assert angle_observer.get_obs(agents['agent0']) == {'ground_angle': {
        'agent0': 123,
        'agent1': 126,
        'agent2': -1,
        'agent3': 46,
        'agent4': -1,
        'agent5': 276,
        'agent7': -1,
        'agent8': 5,
        'agent9': 246
    }}

    assert angle_observer.get_obs(agents['agent1']) == {'ground_angle': {
        'agent0': -1,
        'agent1': 126,
        'agent2': -1,
        'agent3': -1,
        'agent4': -1,
        'agent5': -1,
        'agent7': -1,
        'agent8': 5,
        'agent9': 246
    }}

    assert angle_observer.get_obs(agents['agent9']) == {'ground_angle': {
        'agent0': 123,
        'agent1': 126,
        'agent2': 180,
        'agent3': 46,
        'agent4': 212,
        'agent5': 276,
        'agent7': 0,
        'agent8': 5,
        'agent9': 246
    }}


def test_velocity_observer():
    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=10)
    velocity_state = VelocityState(agents=agents)
    velocity_observer = PositionRestrictedObservationWrapper(VelocityObserver(agents=agents), obs_filter=linear_drop_off, agents=agents)
    
    position_state.reset()
    velocity_state.reset()

    obs = velocity_observer.get_obs(agents['agent0'])['velocity']
    np.testing.assert_array_equal(obs['agent0'], np.array([-0.3, 0.8]))
    np.testing.assert_array_equal(obs['agent1'], np.array([-0.2, 0.7]))
    np.testing.assert_array_equal(obs['agent2'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent3'], np.array([0.0, 0.5]))
    np.testing.assert_array_equal(obs['agent4'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent5'], np.array([0.2, 0.3]))
    np.testing.assert_array_equal(obs['agent6'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent8'], np.array([0.5, 0.0]))
    np.testing.assert_array_equal(obs['agent9'], np.array([0.6, -0.1]))


    obs = velocity_observer.get_obs(agents['agent1'])['velocity']
    np.testing.assert_array_equal(obs['agent0'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent1'], np.array([-0.2, 0.7]))
    np.testing.assert_array_equal(obs['agent2'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent3'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent4'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent5'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent6'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['agent8'], np.array([0.5, 0.0]))
    np.testing.assert_array_equal(obs['agent9'], np.array([0.6, -0.1]))


    obs = velocity_observer.get_obs(agents['agent9'])['velocity']
    np.testing.assert_array_equal(obs['agent0'], np.array([-0.3, 0.8]))
    np.testing.assert_array_equal(obs['agent1'], np.array([-0.2, 0.7]))
    np.testing.assert_array_equal(obs['agent2'], np.array([-0.1, 0.6]))
    np.testing.assert_array_equal(obs['agent3'], np.array([0.0, 0.5]))
    np.testing.assert_array_equal(obs['agent4'], np.array([0.1, 0.4]))
    np.testing.assert_array_equal(obs['agent5'], np.array([0.2, 0.3]))
    np.testing.assert_array_equal(obs['agent6'], np.array([0.3, 0.2]))
    np.testing.assert_array_equal(obs['agent8'], np.array([0.5, 0.0]))
    np.testing.assert_array_equal(obs['agent9'], np.array([0.6, -0.1]))
