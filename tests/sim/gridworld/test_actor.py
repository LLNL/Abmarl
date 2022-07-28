
from gym.spaces import Box, Discrete
import numpy as np
import pytest

from abmarl.sim.gridworld.actor import MoveActor, AttackActor, ActorBaseComponent
from abmarl.sim.gridworld.state import PositionState, HealthState
from abmarl.sim.gridworld.agent import MovingAgent, AttackingAgent, HealthAgent
from abmarl.sim.gridworld.grid import Grid

grid = Grid(5,6)


def test_move_actor():
    agents = {
        'agent0': MovingAgent(
            id='agent0', initial_position=np.array([3, 4]), encoding=1, move_range=1
        ),
        'agent1': MovingAgent(
            id='agent1', initial_position=np.array([2, 2]), encoding=2, move_range=2
        ),
        'agent2': MovingAgent(
            id='agent2', initial_position=np.array([0, 1]), encoding=1, move_range=1
        ),
        'agent3': MovingAgent(
            id='agent3', initial_position=np.array([3, 1]), encoding=3, move_range=3
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    move_actor = MoveActor(grid=grid, agents=agents)
    assert isinstance(move_actor, ActorBaseComponent)
    assert move_actor.key == 'move'
    assert move_actor.supported_agent_type == MovingAgent
    assert agents['agent0'].action_space['move'] == Box(-1, 1, (2,), int)
    assert agents['agent1'].action_space['move'] == Box(-2, 2, (2,), int)
    assert agents['agent2'].action_space['move'] == Box(-1, 1, (2,), int)
    assert agents['agent3'].action_space['move'] == Box(-3, 3, (2,), int)

    for agent in agents.values():
        agent.finalize()
        assert agent.null_action.keys() == set(('move',))
        np.testing.assert_array_equal(agent.null_action['move'], np.zeros((2,), dtype=int))

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


def test_move_actor_with_overlap():
    overlapping = {
        1: [1],
        2: [3],
        3: [2]
    }
    grid = Grid(5, 6, overlapping=overlapping)
    agents = {
        'agent0': MovingAgent(
            id='agent0', initial_position=np.array([4, 4]), encoding=1, move_range=1
        ),
        'agent1': MovingAgent(
            id='agent1', initial_position=np.array([2, 2]), encoding=2, move_range=2
        ),
        'agent2': MovingAgent(
            id='agent2', initial_position=np.array([2, 4]), encoding=1, move_range=1
        ),
        'agent3': MovingAgent(
            id='agent3', initial_position=np.array([3, 2]), encoding=3, move_range=3
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    move_actor = MoveActor(grid=grid, agents=agents)

    position_state.reset()
    action = {
        'agent0': {'move': np.array([-1, 0])},
        'agent1': {'move': np.array([0, 0])},
        'agent2': {'move': np.array([1, 0])},
        'agent3': {'move': np.array([-1, 0])},
    }
    for agent_id, action in action.items():
        move_actor.process_action(agents[agent_id], action)
    np.testing.assert_array_equal(agents['agent0'].position, np.array([3, 4]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([2, 2]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([3, 4]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([2, 2]))

    action = {
        'agent0': {'move': np.array([-1, 0])},
        'agent1': {'move': np.array([0, 2])},
        'agent2': {'move': np.array([0, -1])},
        'agent3': {'move': np.array([1, 1])},
    }
    for agent_id, action in action.items():
        move_actor.process_action(agents[agent_id], action)
    np.testing.assert_array_equal(agents['agent0'].position, np.array([2, 4]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([2, 2]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([3, 3]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([2, 2]))


def test_attack_actor():
    agents = {
        'agent0': HealthAgent(id='agent0', initial_position=np.array([4, 4]), encoding=1),
        'agent1': AttackingAgent(
            id='agent1',
            initial_position=np.array([2, 2]),
            encoding=1,
            attack_range=2,
            attack_strength=1,
            attack_accuracy=1
        ),
        'agent2': HealthAgent(id='agent2', initial_position=np.array([2, 3]), encoding=2),
        'agent3': HealthAgent(id='agent3', initial_position=np.array([3, 2]), encoding=1),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = AttackActor(attack_mapping={1: [1]}, grid=grid, agents=agents)
    assert isinstance(attack_actor, ActorBaseComponent)
    assert attack_actor.key == 'attack'
    assert attack_actor.supported_agent_type == AttackingAgent
    assert agents['agent1'].action_space['attack'] == Discrete(2)

    agents['agent1'].finalize()
    assert agents['agent1'].null_action == {'attack': 0}

    position_state.reset()
    health_state.reset()
    attack_actor.process_action(agents['agent1'], {'attack': 1})
    attack_actor.process_action(agents['agent1'], {'attack': 1})
    assert not agents['agent0'].active
    assert not agents['agent3'].active
    assert agents['agent0'].health <= 0
    assert agents['agent3'].health <= 0
    assert not grid[4, 4]
    assert not grid[3, 2]

    attack_actor.process_action(agents['agent1'], {'attack': 1})
    assert agents['agent2'].active
    assert agents['agent2'].health > 0
    assert grid[2, 3]


def test_attack_actor_attack_mapping():
    agents = {
        'agent0': HealthAgent(id='agent0', initial_position=np.array([4, 4]), encoding=1),
        'agent1': AttackingAgent(
            id='agent1',
            initial_position=np.array([2, 2]),
            encoding=1,
            attack_range=2,
            attack_strength=1,
            attack_accuracy=1
        ),
        'agent2': HealthAgent(id='agent2', initial_position=np.array([2, 3]), encoding=2),
        'agent3': HealthAgent(id='agent3', initial_position=np.array([3, 2]), encoding=1),
    }

    with pytest.raises(AssertionError):
        AttackActor(agents=agents, grid=grid, attack_mapping=[1, 2, 3])

    with pytest.raises(AssertionError):
        AttackActor(agents=agents, grid=grid, attack_mapping={'1': [3], 2.0: [6]})

    with pytest.raises(AssertionError):
        AttackActor(agents=agents, grid=grid, attack_mapping={1: 3, 2: [6]})

    with pytest.raises(AssertionError):
        AttackActor(agents=agents, grid=grid, attack_mapping={1: ['2', 3], 2: [2, 3]})
