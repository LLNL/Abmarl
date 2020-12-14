
from gym.spaces import Box
import numpy as np

from admiral.envs.components.movement import GridMovementAgent, PositionAgent
from admiral.envs.components.movement import GridMovementActor
from admiral.envs.components.position import PositionState

class MovementTestAgent(GridMovementAgent, PositionAgent): pass

def test_grid_movement_component():
    agents = {
        'agent0': MovementTestAgent(id='agent0', starting_position=np.array([6, 4]), move_range=2),
        'agent1': MovementTestAgent(id='agent1', starting_position=np.array([3, 3]), move_range=3),
        'agent2': MovementTestAgent(id='agent2', starting_position=np.array([0, 1]), move_range=1),
        'agent3': MovementTestAgent(id='agent3', starting_position=np.array([8, 4]), move_range=1),
    }
    state = PositionState(region=10, agents=agents)
    actor = GridMovementActor(position=state, agents=agents)
    state.reset()

    actor.process_move(agents['agent0'], np.array([-1,  1]))
    actor.process_move(agents['agent1'], np.array([ 0,  1]))
    actor.process_move(agents['agent2'], np.array([ 0, -1]))
    actor.process_move(agents['agent3'], np.array([ 1,  0]))
    np.testing.assert_array_equal(agents['agent0'].position, np.array([5, 5]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([3, 4]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([9, 4]))

    actor.process_move(agents['agent0'], np.array([ 2, -2]))
    actor.process_move(agents['agent1'], np.array([-3,  0]))
    actor.process_move(agents['agent2'], np.array([-1,  0]))
    actor.process_move(agents['agent3'], np.array([ 1,  1]))
    np.testing.assert_array_equal(agents['agent0'].position, np.array([7, 3]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([0, 4]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([9, 4]))
