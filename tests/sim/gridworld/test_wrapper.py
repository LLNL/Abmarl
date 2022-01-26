
from os import access
from gym.spaces import Discrete, Box
import numpy as np

from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.state import PositionState
from abmarl.sim.gridworld.wrapper import RavelActionWrapper, ActorWrapper, ComponentWrapper
from abmarl.sim.wrappers.ravel_discrete_wrapper import ravel

from .helpers import grid, moving_agents as agents

position_state = PositionState(grid=grid, agents=agents)
move_actor = MoveActor(grid=grid, agents=agents)
ravel_action_wrapper = RavelActionWrapper(move_actor)

def test_ravel_action_wrapper_properties():
    assert ravel_action_wrapper.wrapped_component == move_actor
    assert ravel_action_wrapper.unwrapped == move_actor
    assert ravel_action_wrapper.agents == move_actor.agents
    assert ravel_action_wrapper.grid == move_actor.grid
    assert ravel_action_wrapper.key == move_actor.key
    assert ravel_action_wrapper.supported_agent_type == move_actor.supported_agent_type

def test_ravel_action_wrapper_agent_spaces():
    assert ravel_action_wrapper.from_space['agent0'] == Box(-1, 1, (2,), np.int)
    assert ravel_action_wrapper.agents['agent0'].action_space['move'] == Discrete(9)
    assert ravel_action_wrapper.from_space['agent1'] == Box(-2, 2, (2,), np.int)
    assert ravel_action_wrapper.agents['agent1'].action_space['move'] == Discrete(25)
    assert ravel_action_wrapper.from_space['agent2'] == Box(-1, 1, (2,), np.int)
    assert ravel_action_wrapper.agents['agent2'].action_space['move'] == Discrete(9)
    assert ravel_action_wrapper.from_space['agent3'] == Box(-3, 3, (2,), np.int)
    assert ravel_action_wrapper.agents['agent3'].action_space['move'] == Discrete(49)

def test_ravel_action_wrapper_process_action():
    action_sample = {
        'agent0': {'move': 7},
        'agent1': {'move': 3},
        'agent2': {'move': 4},
        'agent3': {'move': 34},
    }
    np.testing.assert_array_equal(
        ravel_action_wrapper.wrap_point(Box(-1, 1, (2,), np.int), 7),
        np.array([1, 0])
    )
    np.testing.assert_array_equal(
        ravel_action_wrapper.wrap_point(Box(-2, 2, (2,), np.int), 3),
        np.array([-2, 1])
    )
    np.testing.assert_array_equal(
        ravel_action_wrapper.wrap_point(Box(-1, 1, (2,), np.int), 4),
        np.array([0, 0])
    )
    np.testing.assert_array_equal(
        ravel_action_wrapper.wrap_point(Box(-3, 3, (2,), np.int), 34),
        np.array([1, 3])
    )

    position_state.reset()
    assert ravel_action_wrapper.process_action(agents['agent0'], action_sample['agent0'])
    np.testing.assert_array_equal(agents['agent0'].position, np.array([4, 4]))
    assert ravel_action_wrapper.process_action(agents['agent1'], action_sample['agent1'])
    np.testing.assert_array_equal(agents['agent1'].position, np.array([0, 3]))
    assert ravel_action_wrapper.process_action(agents['agent2'], action_sample['agent2'])
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 1]))
    assert not ravel_action_wrapper.process_action(agents['agent3'], action_sample['agent3'])
    np.testing.assert_array_equal(agents['agent3'].position, np.array([3, 1]))
