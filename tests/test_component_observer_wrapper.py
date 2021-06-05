import numpy as np

from admiral.sim.components.agent import SpeedAngleAgent, VelocityAgent, BroadcastingAgent
from admiral.sim.components.state import GridPositionState, LifeState, SpeedAngleState, \
    VelocityState, BroadcastState
from admiral.sim.components.observer import HealthObserver, LifeObserver, PositionObserver, \
    RelativePositionObserver, SpeedObserver, AngleObserver, VelocityObserver, TeamObserver
from admiral.sim.components.wrappers.observer_wrapper import \
    PositionRestrictedObservationWrapper, TeamBasedCommunicationWrapper
from admiral.sim.components.actor import BroadcastActor

from admiral.sim.components.agent import AgentObservingAgent, VelocityObservingAgent, \
    PositionObservingAgent, SpeedAngleObservingAgent, TeamObservingAgent, LifeObservingAgent, \
    HealthObservingAgent


class AllObservingAgent(
    AgentObservingAgent, VelocityObservingAgent, PositionObservingAgent, SpeedAngleObservingAgent,
    TeamObservingAgent, LifeObservingAgent, HealthObservingAgent
): pass


class NonViewAgent(
    VelocityObservingAgent, PositionObservingAgent, SpeedAngleObservingAgent, TeamObservingAgent,
    LifeObservingAgent, HealthObservingAgent
): pass


class AllAgent(           AllObservingAgent, SpeedAngleAgent, VelocityAgent): pass
class SpeedAnglelessAgent(AllObservingAgent,                  VelocityAgent): pass
class VelocitylessAgent(  AllObservingAgent, SpeedAngleAgent               ): pass


def test_position_restricted_observer_wrapper():
    agents = {
        'agent0': AllAgent(
            id='agent0', agent_view=2, initial_position=np.array([2, 2]), initial_health=0.67,
            team=1, max_speed=1, initial_speed=0.30, initial_banking_angle=7,
            initial_ground_angle=123, initial_velocity=np.array([-0.3, 0.8])
        ),
        'agent1': AllAgent(
            id='agent1', agent_view=1, initial_position=np.array([4, 4]), initial_health=0.54,
            team=2, max_speed=2, initial_speed=0.00, initial_banking_angle=0,
            initial_ground_angle=126, initial_velocity=np.array([-0.2, 0.7])
        ),
        'agent2': AllAgent(
            id='agent2', agent_view=1, initial_position=np.array([4, 3]), initial_health=0.36,
            team=2, max_speed=1, initial_speed=0.12, initial_banking_angle=30,
            initial_ground_angle=180, initial_velocity=np.array([-0.1, 0.6])
        ),
        'agent3': AllAgent(
            id='agent3', agent_view=1, initial_position=np.array([4, 2]), initial_health=0.24,
            team=2, max_speed=4, initial_speed=0.05, initial_banking_angle=13,
            initial_ground_angle=46,  initial_velocity=np.array([0.0,  0.5])
        ),
        'agent6': SpeedAnglelessAgent(
            id='agent6', agent_view=0, initial_position=np.array([1, 1]), initial_health=0.53,
            team=1, max_speed=1, initial_velocity=np.array([0.3, 0.2])
        ),
        'agent7': VelocitylessAgent(
            id='agent7', agent_view=5, initial_position=np.array([0, 4]), initial_health=0.50,
            team=2, max_speed=1, initial_speed=0.36, initial_banking_angle=24,
            initial_ground_angle=0
        ),
    }

    def linear_drop_off(distance, view):
        return 1. - 1. / (view+1) * distance

    np.random.seed(12)
    position_state = GridPositionState(agents=agents, region=5)
    life_state = LifeState(agents=agents)
    speed_state = SpeedAngleState(agents=agents)
    angle_state = SpeedAngleState(agents=agents)
    velocity_state = VelocityState(agents=agents)

    position_observer = PositionObserver(position_state=position_state, agents=agents)
    relative_position_observer = RelativePositionObserver(
        position_state=position_state, agents=agents
    )
    health_observer = HealthObserver(agents=agents)
    life_observer = LifeObserver(agents=agents)
    team_observer = TeamObserver(number_of_teams=3, agents=agents)
    speed_observer = SpeedObserver(agents=agents)
    angle_observer = AngleObserver(agents=agents)
    velocity_observer = VelocityObserver(agents=agents)

    position_state.reset()
    life_state.reset()
    speed_state.reset()
    angle_state.reset()
    velocity_state.reset()

    wrapped_observer = PositionRestrictedObservationWrapper(
        [
            position_observer,
            relative_position_observer,
            health_observer,
            life_observer,
            team_observer,
            speed_observer,
            angle_observer,
            velocity_observer
        ],
        obs_filter=linear_drop_off,
        agents=agents
    )

    obs = wrapped_observer.get_obs(agents['agent0'])
    assert obs['health'] == {
        'agent0': 0.67,
        'agent1': -1,
        'agent2': 0.36,
        'agent3': -1,
        'agent6': 0.53,
        'agent7': -1,
    }
    assert obs['life'] == {
        'agent0': True,
        'agent1': -1,
        'agent2': True,
        'agent3': -1,
        'agent6': 1,
        'agent7': -1,
    }
    assert obs['team'] == {
        'agent0': 1,
        'agent1': -1,
        'agent2': 2,
        'agent3': -1,
        'agent6': 1,
        'agent7': -1,
    }
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([2, 2]))
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([4, 3]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent6'], np.array([1, 1]))
    np.testing.assert_array_equal(obs['position']['agent7'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['relative_position']['agent0'], np.array([0, 0]))
    np.testing.assert_array_equal(obs['relative_position']['agent1'], np.array([-5, -5]))
    np.testing.assert_array_equal(obs['relative_position']['agent2'], np.array([2, 1]))
    np.testing.assert_array_equal(obs['relative_position']['agent3'], np.array([-5, -5]))
    np.testing.assert_array_equal(obs['relative_position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['relative_position']['agent7'], np.array([-5, -5]))
    assert obs['mask'] == {
        'agent0': 1,
        'agent1': 0,
        'agent2': 1,
        'agent3': 0,
        'agent6': 1,
        'agent7': 0,
    }
    assert obs['speed'] == {
        'agent0': 0.3,
        'agent1': -1,
        'agent2': 0.12,
        'agent3': -1,
        'agent6': -1,
        'agent7': -1,
    }
    assert obs['ground_angle'] == {
        'agent0': 123,
        'agent1': -1,
        'agent2': 180,
        'agent3': -1,
        'agent6': -1,
        'agent7': -1,
    }
    np.testing.assert_array_equal(obs['velocity']['agent0'], np.array([-0.3, 0.8]))
    np.testing.assert_array_equal(obs['velocity']['agent1'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['velocity']['agent2'], np.array([-0.1, 0.6]))
    np.testing.assert_array_equal(obs['velocity']['agent3'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['velocity']['agent6'], np.array([0.3, 0.2]))
    np.testing.assert_array_equal(obs['velocity']['agent7'], np.array([0.0, 0.0]))


    obs = wrapped_observer.get_obs(agents['agent1'])
    assert obs['health'] == {
        'agent0': -1,
        'agent1': 0.54,
        'agent2': -1,
        'agent3': -1,
        'agent6': -1,
        'agent7': -1,
    }
    assert obs['life'] == {
        'agent0': -1,
        'agent1': True,
        'agent2': -1,
        'agent3': -1,
        'agent6': -1,
        'agent7': -1,
    }
    assert obs['team'] == {
        'agent0': -1,
        'agent1': 2,
        'agent2': -1,
        'agent3': -1,
        'agent6': -1,
        'agent7': -1,
    }
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([4, 4]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent6'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent7'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['relative_position']['agent0'], np.array([-5, -5]))
    np.testing.assert_array_equal(obs['relative_position']['agent1'], np.array([0, 0]))
    np.testing.assert_array_equal(obs['relative_position']['agent2'], np.array([-5, -5]))
    np.testing.assert_array_equal(obs['relative_position']['agent3'], np.array([-5, -5]))
    np.testing.assert_array_equal(obs['relative_position']['agent6'], np.array([-5, -5]))
    np.testing.assert_array_equal(obs['relative_position']['agent7'], np.array([-5, -5]))
    assert obs['mask'] == {
        'agent0': 0,
        'agent1': 1,
        'agent2': 0,
        'agent3': 0,
        'agent6': 0,
        'agent7': 0,
    }
    assert obs['speed'] == {
        'agent0': -1,
        'agent1': 0.0,
        'agent2': -1,
        'agent3': -1,
        'agent6': -1,
        'agent7': -1,
    }
    assert obs['ground_angle'] == {
        'agent0': -1,
        'agent1': 126,
        'agent2': -1,
        'agent3': -1,
        'agent6': -1,
        'agent7': -1,
    }
    np.testing.assert_array_equal(obs['velocity']['agent0'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['velocity']['agent1'], np.array([-0.2, 0.7]))
    np.testing.assert_array_equal(obs['velocity']['agent2'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['velocity']['agent3'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['velocity']['agent6'], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs['velocity']['agent7'], np.array([0.0, 0.0]))


class CommunicatingAgent(
    BroadcastingAgent, PositionObservingAgent, TeamObservingAgent, AgentObservingAgent
): pass


def test_broadcast_communication_observer_wrapper():
    agents = {
        'agent0': CommunicatingAgent(
            id='agent0', initial_position=np.array([1, 7]), team=1, broadcast_range=0,
            agent_view=0
        ),
        'agent1': CommunicatingAgent(
            id='agent1', initial_position=np.array([3, 3]), team=1, broadcast_range=4,
            agent_view=3
        ),
        'agent2': CommunicatingAgent(
            id='agent2', initial_position=np.array([5, 0]), team=2, broadcast_range=4,
            agent_view=2
        ),
        'agent3': CommunicatingAgent(
            id='agent3', initial_position=np.array([6, 9]), team=2, broadcast_range=4,
            agent_view=2
        ),
        'agent4': CommunicatingAgent(
            id='agent4', initial_position=np.array([4, 7]), team=2, broadcast_range=4,
            agent_view=3
        ),
    }

    position_state = GridPositionState(region=10, agents=agents)
    broadcast_state = BroadcastState(agents=agents)

    position_observer = PositionObserver(position_state=position_state, agents=agents)
    team_observer = TeamObserver(number_of_teams=2, agents=agents)
    partial_observer = PositionRestrictedObservationWrapper(
        [position_observer, team_observer], agents=agents
    )
    comms_observer = TeamBasedCommunicationWrapper([partial_observer], agents=agents)

    broadcast_actor = BroadcastActor(broadcast_state=broadcast_state, agents=agents)

    position_state.reset()
    broadcast_state.reset()

    action_dict = {
        'agent0': {'broadcast': 0},
        'agent1': {'broadcast': 1},
        'agent2': {'broadcast': 0},
        'agent3': {'broadcast': 0},
        'agent4': {'broadcast': 1},
    }
    for agent_id, action in action_dict.items():
        broadcast_actor.process_action(agents[agent_id], action)

    obs = partial_observer.get_obs(agents['agent0'])
    assert obs['mask'] == {
        'agent0': 1,
        'agent1': 0,
        'agent2': 0,
        'agent3': 0,
        'agent4': 0,
    }
    assert obs['team'] == {
        'agent0': 1,
        'agent1': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': -1,
    }
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([1, 7]))
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent4'], np.array([-1, -1]))

    obs = comms_observer.get_obs(agents['agent0'])
    assert obs['mask'] == {
        'agent0': 1,
        'agent1': 1,
        'agent2': 1,
        'agent3': 0,
        'agent4': 1,
    }
    assert obs['team'] == {
        'agent0': 1,
        'agent1': 1,
        'agent2': 2,
        'agent3': -1,
        'agent4': 2,
    }
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([1, 7]))
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([3, 3]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([5, 0]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent4'], np.array([4, 7]))


    action_dict = {
        'agent0': {'broadcast': 0},
        'agent1': {'broadcast': 0},
        'agent2': {'broadcast': 1},
        'agent3': {'broadcast': 1},
        'agent4': {'broadcast': 0},
    }
    for agent_id, action in action_dict.items():
        broadcast_actor.process_action(agents[agent_id], action)

    obs = comms_observer.get_obs(agents['agent0'])
    assert obs['mask'] == {
        'agent0': 1,
        'agent1': 0,
        'agent2': 0,
        'agent3': 0,
        'agent4': 0,
    }
    assert obs['team'] == {
        'agent0': 1,
        'agent1': -1,
        'agent2': -1,
        'agent3': -1,
        'agent4': -1,
    }
    np.testing.assert_array_equal(obs['position']['agent0'], np.array([1, 7]))
    np.testing.assert_array_equal(obs['position']['agent1'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent2'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent3'], np.array([-1, -1]))
    np.testing.assert_array_equal(obs['position']['agent4'], np.array([-1, -1]))
