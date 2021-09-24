
import numpy as np

from abmarl.sim.gridworld.actor import MoveActor, AttackActor, ActorBaseComponent
from abmarl.sim.gridworld.state import PositionState, HealthState
from abmarl.sim.gridworld.agent import MovingAgent, AttackingAgent
from abmarl.sim.gridworld.grid import Grid

def test_move_actor():
    grid = Grid(5, 6)
    agents = {
        'agent0': MovingAgent(id='agent0', initial_position=np.array([3, 4]), encoding=1, move_range=1),
        'agent1': MovingAgent(id='agent1', initial_position=np.array([2, 2]), encoding=2, move_range=2),
        'agent2': MovingAgent(id='agent2', initial_position=np.array([0, 1]), encoding=1, move_range=1),
        'agent3': MovingAgent(id='agent3', initial_position=np.array([3, 1]), encoding=3, move_range=3),
    }

    position_state = PositionState(grid=grid, agents=agents)
    move_actor = MoveActor(grid=grid, agents=agents)

    position_state.reset()
    action = {
        'agent0': {'move': np.array([1, 1])},
        'agent1': {'move': np.array([-1, 0])},
        'agent2': {'move': np.array([0, 1])},
        'agent3': {'move': np.array([-1, 1])},
    }
    for agent_id, action in action.items():
        move_actor.process_action(agents[agent_id], action)
    np.testing.assert_array_equal(agents['agent0'].position, np.array([4, 5]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([1, 2]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 2]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([2, 2]))

    action = {
        'agent0': {'move': np.array([1, 1])},
        'agent1': {'move': np.array([0, 0])},
        'agent2': {'move': np.array([-1, 1])},
        'agent3': {'move': np.array([-1, 0])},
    }
    for agent_id, action in action.items():
        move_actor.process_action(agents[agent_id], action)
    np.testing.assert_array_equal(agents['agent0'].position, np.array([4, 5]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([1, 2]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 2]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([2, 2]))
