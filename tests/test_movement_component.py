
from gym.spaces import Box
import numpy as np

from admiral.component_envs.movement import GridMovementAgent, GridPositionAgent
from admiral.component_envs.movement import GridMovementComponent

class MovementTestAgent(GridMovementAgent, GridPositionAgent): pass

def test_grid_movement_component():
    agents = {
        'agent0': MovementTestAgent(id='agent0', starting_position=np.array([6, 4]), move = 2),
        'agent1': MovementTestAgent(id='agent1', starting_position=np.array([3, 3]), move = 3),
        'agent2': MovementTestAgent(id='agent2', starting_position=np.array([0, 1]), move = 1),
        'agent3': MovementTestAgent(id='agent3', starting_position=np.array([8, 4]), move = 1),
    }
    component = GridMovementComponent(agents=agents, region=10)
    for agent in agents.values():
        agent.position = agent.starting_position
        assert agent.action_space['move'] == Box(-agent.move, agent.move, (2,), np.int)

    component.process_move(agents['agent0'], np.array([-1,  1]))
    component.process_move(agents['agent1'], np.array([ 0,  1]))
    component.process_move(agents['agent2'], np.array([ 0, -1]))
    component.process_move(agents['agent3'], np.array([ 1,  0]))
    np.testing.assert_array_equal(agents['agent0'].position, np.array([5, 5]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([3, 4]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([9, 4]))

    component.process_move(agents['agent0'], np.array([ 2, -2]))
    component.process_move(agents['agent1'], np.array([-3,  0]))
    component.process_move(agents['agent2'], np.array([-1,  0]))
    component.process_move(agents['agent3'], np.array([ 1,  1]))
    np.testing.assert_array_equal(agents['agent0'].position, np.array([7, 3]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([0, 4]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([9, 4]))
