import numpy as np

from admiral.sim.components.state import VelocityState, ContinuousPositionState
from admiral.sim.components.actor import AccelerationMovementActor, ContinuousCollisionActor
from admiral.sim.components.actor import VelocityAgent, AcceleratingAgent, CollisionAgent


class ParticleAgent(VelocityAgent, AcceleratingAgent, CollisionAgent): pass


def test_collision():

    agents = {
        'agent0': ParticleAgent(
            id='agent0', max_acceleration=0, max_speed=10, size=1, mass=1,
            initial_velocity=np.array([1, 1]), initial_position=np.array([1,1])
        ),
        'agent1': ParticleAgent(
            id='agent1', max_acceleration=0, max_speed=10, size=1, mass=1,
            initial_velocity=np.array([-1, 1]), initial_position=np.array([4, 1])
        )
    }

    position_state = ContinuousPositionState(region=10, agents=agents)
    velocity_state = VelocityState(agents=agents, friction=0.0)
    position_state.reset()
    velocity_state.reset()

    movement_actor = AccelerationMovementActor(
        position_state=position_state, velocity_state=velocity_state, agents=agents
    )
    collision_actor = ContinuousCollisionActor(
        position_state=position_state, velocity_state=velocity_state, agents=agents
    )

    np.testing.assert_array_equal(
        movement_actor.process_action(agents['agent0'], {'accelerate': np.zeros(2)}),
        np.array([1., 1.])
    )
    np.testing.assert_array_equal(agents['agent0'].position, np.array([2., 2.]))
    np.testing.assert_array_equal(agents['agent0'].velocity, np.array([1., 1.]))
    np.testing.assert_array_equal(
        movement_actor.process_action(agents['agent1'], {'accelerate': np.zeros(2)}),
        np.array([-1., 1.])
    )
    np.testing.assert_array_equal(agents['agent1'].position, np.array([3., 2.]))
    np.testing.assert_array_equal(agents['agent1'].velocity, np.array([-1., 1.]))

    collision_actor.detect_collisions_and_modify_states()
    np.testing.assert_array_equal(agents['agent0'].position, np.array([1.5, 1.5]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([3.5, 1.5]))
    np.testing.assert_array_equal(agents['agent0'].velocity, np.array([-1., 1.]))
    np.testing.assert_array_equal(agents['agent1'].velocity, np.array([1., 1.]))

    np.testing.assert_array_equal(
        movement_actor.process_action(agents['agent0'], {'accelerate': np.zeros(2)}),
        np.array([-1., 1.])
    )
    np.testing.assert_array_equal(
        movement_actor.process_action(agents['agent1'], {'accelerate': np.zeros(2)}),
        np.array([1., 1.])
    )
