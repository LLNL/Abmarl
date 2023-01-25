
from gym.spaces import Discrete, Box, Dict
import numpy as np
import pytest

from abmarl.sim.gridworld.actor import MoveActor, EncodingBasedAttackActor, ActorBaseComponent
from abmarl.sim.gridworld.state import PositionState, HealthState
from abmarl.sim.gridworld.wrapper import RavelActionWrapper, ExclusiveChannelActionWrapper, \
    ActorWrapper
from abmarl.sim.gridworld.grid import Grid
from abmarl.sim.gridworld.agent import MovingAgent, HealthAgent, AttackingAgent


class WrappingTestAgent(MovingAgent, HealthAgent, AttackingAgent):
    def __init__(self, **kwargs):
        super().__init__(
            attack_strength=0.5,
            attack_accuracy=1,
            attack_range=4,
            **kwargs
        )


grid = Grid(5,6)

agents = {
    'agent0': WrappingTestAgent(
        id='agent0', initial_position=np.array([3, 4]), encoding=1, move_range=1
    ),
    'agent1': WrappingTestAgent(
        id='agent1', initial_position=np.array([2, 2]), encoding=2, move_range=2
    ),
    'agent2': WrappingTestAgent(
        id='agent2', initial_position=np.array([0, 1]), encoding=1, move_range=1
    ),
    'agent3': WrappingTestAgent(
        id='agent3', initial_position=np.array([3, 1]), encoding=3, move_range=3
    ),
}


position_state = PositionState(grid=grid, agents=agents)
health_state = HealthState(grid=grid, agents=agents)
move_actor = MoveActor(grid=grid, agents=agents)
attack_actor = EncodingBasedAttackActor(
    grid=grid,
    agents=agents,
    attack_mapping={1: [1, 2, 3], 2: [1, 3], 3: [1]}
)
ravelled_move_actor = RavelActionWrapper(move_actor)
exclusive_attack_actor = ExclusiveChannelActionWrapper(attack_actor)


def test_ravelled_move_wrapper_properties():
    assert isinstance(ravelled_move_actor, ActorWrapper)
    assert isinstance(ravelled_move_actor, ActorBaseComponent)
    assert ravelled_move_actor.wrapped_component == move_actor
    assert ravelled_move_actor.unwrapped == move_actor
    assert ravelled_move_actor.agents == move_actor.agents
    assert ravelled_move_actor.grid == move_actor.grid
    assert ravelled_move_actor.key == move_actor.key
    assert ravelled_move_actor.supported_agent_type == move_actor.supported_agent_type


def test_ravelled_move_wrapper_agent_spaces():
    assert ravelled_move_actor.from_space['agent0'] == Box(-1, 1, (2,), int)
    assert ravelled_move_actor.agents['agent0'].action_space['move'] == Discrete(9)
    assert ravelled_move_actor.from_space['agent1'] == Box(-2, 2, (2,), int)
    assert ravelled_move_actor.agents['agent1'].action_space['move'] == Discrete(25)
    assert ravelled_move_actor.from_space['agent2'] == Box(-1, 1, (2,), int)
    assert ravelled_move_actor.agents['agent2'].action_space['move'] == Discrete(9)
    assert ravelled_move_actor.from_space['agent3'] == Box(-3, 3, (2,), int)
    assert ravelled_move_actor.agents['agent3'].action_space['move'] == Discrete(49)


def test_ravelled_move_wrapper_agent_null_action():
    assert agents['agent0'].null_action['move'] == 4
    np.testing.assert_array_equal(
        ravelled_move_actor.wrap_point(
            ravelled_move_actor.from_space['agent0'], agents['agent0'].null_action['move']
        ),
        np.array([0, 0], dtype=int)
    )

    assert agents['agent1'].null_action['move'] == 12
    np.testing.assert_array_equal(
        ravelled_move_actor.wrap_point(
            ravelled_move_actor.from_space['agent1'], agents['agent1'].null_action['move']
        ),
        np.array([0, 0], dtype=int)
    )

    assert agents['agent2'].null_action['move'] == 4
    np.testing.assert_array_equal(
        ravelled_move_actor.wrap_point(
            ravelled_move_actor.from_space['agent2'], agents['agent2'].null_action['move']
        ),
        np.array([0, 0], dtype=int)
    )

    assert agents['agent3'].null_action['move'] == 24
    np.testing.assert_array_equal(
        ravelled_move_actor.wrap_point(
            ravelled_move_actor.from_space['agent2'], agents['agent2'].null_action['move']
        ),
        np.array([0, 0], dtype=int)
    )


def test_ravelled_move_wrapper_process_action():
    action_sample = {
        'agent0': {'move': 7},
        'agent1': {'move': 3},
        'agent2': {'move': 4},
        'agent3': {'move': 34},
    }
    np.testing.assert_array_equal(
        ravelled_move_actor.wrap_point(Box(-1, 1, (2,), int), 7),
        np.array([1, 0])
    )
    np.testing.assert_array_equal(
        ravelled_move_actor.wrap_point(Box(-2, 2, (2,), int), 3),
        np.array([-2, 1])
    )
    np.testing.assert_array_equal(
        ravelled_move_actor.wrap_point(Box(-1, 1, (2,), int), 4),
        np.array([0, 0])
    )
    np.testing.assert_array_equal(
        ravelled_move_actor.wrap_point(Box(-3, 3, (2,), int), 34),
        np.array([1, 3])
    )

    position_state.reset()
    health_state.reset()
    assert ravelled_move_actor.process_action(agents['agent0'], action_sample['agent0'])
    np.testing.assert_array_equal(agents['agent0'].position, np.array([4, 4]))
    assert ravelled_move_actor.process_action(agents['agent1'], action_sample['agent1'])
    np.testing.assert_array_equal(agents['agent1'].position, np.array([0, 3]))
    assert ravelled_move_actor.process_action(agents['agent2'], action_sample['agent2'])
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 1]))
    assert not ravelled_move_actor.process_action(agents['agent3'], action_sample['agent3'])
    np.testing.assert_array_equal(agents['agent3'].position, np.array([3, 1]))


def test_exclusive_attack_wrapper_non_dict_space():
    with pytest.raises(AssertionError):
        # This will fail because the space is Discrete, not Dict
        ExclusiveChannelActionWrapper(ravelled_move_actor)


def test_exclusive_attack_wrapper_properties():
    assert isinstance(exclusive_attack_actor, ActorWrapper)
    assert isinstance(exclusive_attack_actor, ActorBaseComponent)
    assert exclusive_attack_actor.wrapped_component == attack_actor
    assert exclusive_attack_actor.unwrapped == attack_actor
    assert exclusive_attack_actor.agents == attack_actor.agents
    assert exclusive_attack_actor.grid == attack_actor.grid
    assert exclusive_attack_actor.key == attack_actor.key
    assert exclusive_attack_actor.supported_agent_type == attack_actor.supported_agent_type


def test_exclusive_attack_wrapper_agent_spaces():
    assert exclusive_attack_actor.from_space['agent0'] == Dict(
        {1: Discrete(2), 2: Discrete(2), 3: Discrete(2)}
    )
    assert exclusive_attack_actor.agents['agent0'].action_space['attack'] == Discrete(4)
    assert exclusive_attack_actor.from_space['agent1'] == Dict({1: Discrete(2), 3: Discrete(2)})
    assert exclusive_attack_actor.agents['agent1'].action_space['attack'] == Discrete(3)
    assert exclusive_attack_actor.from_space['agent2'] == Dict(
        {1: Discrete(2), 2: Discrete(2), 3: Discrete(2)}
    )
    assert exclusive_attack_actor.agents['agent2'].action_space['attack'] == Discrete(4)
    assert exclusive_attack_actor.from_space['agent3'] == Dict({1: Discrete(2)})
    assert exclusive_attack_actor.agents['agent3'].action_space['attack'] == Discrete(2)


def test_exclusive_attack_wrapper_agent_null_actions():
    assert agents['agent0'].null_action['attack'] == 0
    assert exclusive_attack_actor.wrap_point(
        exclusive_attack_actor.from_space['agent0'], agents['agent0'].null_action['attack']
    ) == {1: 0, 2: 0, 3: 0}

    assert agents['agent1'].null_action['attack'] == 0
    assert exclusive_attack_actor.wrap_point(
        exclusive_attack_actor.from_space['agent1'], agents['agent1'].null_action['attack']
    ) == {1: 0, 3: 0}

    assert agents['agent2'].null_action['attack'] == 0
    assert exclusive_attack_actor.wrap_point(
        exclusive_attack_actor.from_space['agent2'], agents['agent2'].null_action['attack']
    ) == {1: 0, 2: 0, 3: 0}

    assert agents['agent3'].null_action['attack'] == 0
    assert exclusive_attack_actor.wrap_point(
        exclusive_attack_actor.from_space['agent3'], agents['agent3'].null_action['attack']
    ) == {1: 0}


def test_exclusive_attack_wrapper_process_action():
    action_sample = {
        'agent0': {'attack': 0},
        'agent1': {'attack': 2},
        'agent2': {'attack': 2},
        'agent3': {'attack': 1},
    }
    assert exclusive_attack_actor.wrap_point(
        Dict({1: Discrete(2), 2: Discrete(2), 3: Discrete(2)}),
        0
    ) == {1: 0, 2: 0, 3: 0}
    assert exclusive_attack_actor.wrap_point(
        Dict({1: Discrete(2), 3: Discrete(2)}),
        2
    ) == {1: 0, 3: 1}
    assert exclusive_attack_actor.wrap_point(
        Dict({1: Discrete(2), 2: Discrete(2), 3: Discrete(2)}),
        2
    ) == {1: 0, 2: 1, 3: 0}
    assert exclusive_attack_actor.wrap_point(
        Dict({1: Discrete(2)}),
        1
    ) == {1: 1}

    position_state.reset()
    health_state.reset()
    attack_status, attacked_agents = \
        exclusive_attack_actor.process_action(agents['agent0'], action_sample['agent0'])
    assert not attack_status
    assert not attacked_agents
    attack_status, attacked_agents = \
        exclusive_attack_actor.process_action(agents['agent1'], action_sample['agent1'])
    assert attack_status
    assert len(attacked_agents) == 1
    assert attacked_agents[0].encoding == 3
    attack_status, attacked_agents = \
        exclusive_attack_actor.process_action(agents['agent2'], action_sample['agent2'])
    assert attack_status
    assert len(attacked_agents) == 1
    assert attacked_agents[0].encoding == 2
    attack_status, attacked_agents = \
        exclusive_attack_actor.process_action(agents['agent3'], action_sample['agent3'])
    assert attack_status
    assert len(attacked_agents) == 1
    assert attacked_agents[0].encoding == 1
