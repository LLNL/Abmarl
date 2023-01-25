
from gym.spaces import Box, Discrete, MultiDiscrete, Dict
import numpy as np
import pytest

from abmarl.sim.gridworld.actor import MoveActor, BinaryAttackActor, SelectiveAttackActor, \
    EncodingBasedAttackActor, RestrictedSelectiveAttackActor, ActorBaseComponent
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


def test_binary_attack_actor():
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
    attack_actor = BinaryAttackActor(attack_mapping={1: [1]}, grid=grid, agents=agents)
    assert isinstance(attack_actor, ActorBaseComponent)
    assert attack_actor.key == 'attack'
    assert attack_actor.supported_agent_type == AttackingAgent
    assert agents['agent1'].action_space['attack'] == Discrete(2)

    agents['agent1'].finalize()
    assert agents['agent1'].null_action == {'attack': 0}

    position_state.reset()
    health_state.reset()
    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], {'attack': 1})
    assert attack_status
    assert attacked_agents
    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], {'attack': 1})
    assert attack_status
    assert attacked_agents
    assert not agents['agent0'].active
    assert not agents['agent3'].active
    assert agents['agent0'].health <= 0
    assert agents['agent3'].health <= 0
    assert not grid[4, 4]
    assert not grid[3, 2]

    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], {'attack': 1})
    assert attack_status
    assert not attacked_agents
    assert agents['agent2'].active
    assert agents['agent2'].health > 0
    assert grid[2, 3]


def test_binary_attack_actor_attack_mapping():
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
        BinaryAttackActor(agents=agents, grid=grid, attack_mapping=[1, 2, 3])

    with pytest.raises(AssertionError):
        BinaryAttackActor(agents=agents, grid=grid, attack_mapping={'1': [3], 2.0: [6]})

    with pytest.raises(AssertionError):
        BinaryAttackActor(agents=agents, grid=grid, attack_mapping={1: 3, 2: [6]})

    with pytest.raises(AssertionError):
        BinaryAttackActor(agents=agents, grid=grid, attack_mapping={1: ['2', 3], 2: [2, 3]})


def test_binary_attack_actor_attack_count():
    agents = {
        'agent0': AttackingAgent(
            id='agent0',
            initial_position=np.array([2, 2]),
            encoding=1,
            attack_range=2,
            attack_strength=0,
            attack_accuracy=1,
            attack_count=3
        ),
        'agent1': HealthAgent(
            id='agent1', initial_position=np.array([4, 4]), encoding=1, initial_health=1
        ),
        'agent2': HealthAgent(
            id='agent2', initial_position=np.array([2, 3]), encoding=2, initial_health=1
        ),
        'agent3': HealthAgent(
            id='agent3', initial_position=np.array([3, 2]), encoding=1, initial_health=1
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = BinaryAttackActor(attack_mapping={1: [1, 2]}, grid=grid, agents=agents)
    assert agents['agent0'].action_space['attack'] == Discrete(4)

    agents['agent0'].finalize()
    assert agents['agent0'].null_action == {'attack': 0}

    position_state.reset()
    health_state.reset()
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 0})
    assert not attack_status
    assert not attacked_agents

    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 1})
    assert attack_status
    assert len(attacked_agents) == 1

    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 2})
    assert attack_status
    assert len(attacked_agents) == 2

    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 3})
    assert attack_status
    assert len(attacked_agents) == 3

    agents['agent0'].attack_strength = 1
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 3})
    assert attack_status
    assert len(attacked_agents) == 3
    assert not agents['agent1'].active
    assert not agents['agent2'].active
    assert not agents['agent3'].active
    assert agents['agent1'].health <= 0
    assert agents['agent2'].health <= 0
    assert agents['agent3'].health <= 0
    assert not grid[4, 4]
    assert not grid[2, 3]
    assert not grid[3, 2]


def test_binary_attack_actor_stacked_attack():
    agents = {
        'agent0': AttackingAgent(
            id='agent0',
            initial_position=np.array([2, 2]),
            encoding=1,
            attack_range=2,
            attack_strength=1,
            attack_accuracy=1,
            attack_count=2
        ),
        'agent1': HealthAgent(
            id='agent1', initial_position=np.array([4, 4]), encoding=1, initial_health=1
        ),
        'agent2': HealthAgent(
            id='agent2', initial_position=np.array([2, 3]), encoding=2, initial_health=1
        ),
        'agent3': HealthAgent(
            id='agent3', initial_position=np.array([3, 2]), encoding=1, initial_health=1
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = BinaryAttackActor(
        attack_mapping={1: [1]}, stacked_attacks=False, grid=grid, agents=agents
    )
    assert agents['agent0'].action_space['attack'] == Discrete(3)

    agents['agent0'].finalize()
    assert agents['agent0'].null_action == {'attack': 0}

    position_state.reset()
    health_state.reset()
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 2})
    assert attack_status
    assert len(attacked_agents) == 2
    assert agents['agent1'] in attacked_agents
    assert agents['agent3'] in attacked_agents
    assert agents['agent2'] not in attacked_agents
    assert not agents['agent1'].active
    assert not agents['agent3'].active
    assert agents['agent1'].health <= 0
    assert agents['agent3'].health <= 0
    assert not grid[4, 4]
    assert not grid[3, 2]

    agents['agent0'].attack_strength = 0.5
    attack_actor = BinaryAttackActor(
        attack_mapping={1: [2]}, stacked_attacks=True, grid=grid, agents=agents
    )

    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 1})
    assert attack_status
    assert len(attacked_agents) == 1
    assert attacked_agents[0] == agents['agent2']
    assert agents['agent2'].health == 0.5

    agents['agent0'].attack_strength = 0
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 2})
    assert attack_status
    assert len(attacked_agents) == 2
    assert attacked_agents[0] == attacked_agents[1]
    assert agents['agent2'].health == 0.5

    agents['agent0'].attack_strength = 0.25
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], {'attack': 2})
    assert attack_status
    assert len(attacked_agents) == 2
    assert attacked_agents[0] == attacked_agents[1]
    assert not agents['agent2'].active
    assert agents['agent2'].health <= 0
    assert not grid[2, 3]


def test_selective_attack_actor():
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
    attack_actor = SelectiveAttackActor(attack_mapping={1: [1]}, grid=grid, agents=agents)
    assert isinstance(attack_actor, ActorBaseComponent)
    assert attack_actor.key == 'attack'
    assert attack_actor.supported_agent_type == AttackingAgent
    assert agents['agent1'].action_space['attack'] == Box(0, 1, (5, 5), int)

    agents['agent1'].finalize()
    np.testing.assert_array_equal(
        agents['agent1'].null_action['attack'],
        np.zeros((5,5), dtype=int)
    )

    position_state.reset()
    health_state.reset()

    # Attacking agent0
    attack = {'attack': np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ], dtype=int)}
    assert attack in agents['agent1'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], attack)
    assert attack_status
    assert attacked_agents == [agents['agent0']]
    assert not agents['agent0'].active
    assert agents['agent0'].health <= 0
    assert not grid[4, 4]

    # Attacking agent3
    attack = {'attack': np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent1'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], attack)
    assert attack_status
    assert attacked_agents == [agents['agent3']]
    assert not agents['agent3'].active
    assert agents['agent3'].health <= 0
    assert not grid[3, 2]


    # Attacking both agent0 and agent3
    position_state.reset()
    health_state.reset()

    attack = {'attack': np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1]
    ], dtype=int)}
    assert attack in agents['agent1'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], attack)
    assert attack_status
    assert attacked_agents == [agents['agent3'], agents['agent0']]
    assert not agents['agent0'].active
    assert not agents['agent3'].active
    assert agents['agent0'].health <= 0
    assert agents['agent3'].health <= 0
    assert not grid[4, 4]
    assert not grid[3, 2]


    # Attacking everywhere except for agent0 and agent3
    position_state.reset()
    health_state.reset()

    attack = {'attack': np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0]
    ], dtype=int)}
    assert attack in agents['agent1'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], attack)
    assert attack_status
    assert not attacked_agents
    assert agents['agent0'].active
    assert agents['agent1'].active
    assert agents['agent2'].active
    assert agents['agent3'].active
    assert grid[4, 4]
    assert grid[3, 2]
    assert grid[2, 2]
    assert grid[2, 3]


    # Attacking everywhere
    attack = {'attack': np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ], dtype=int)}
    assert attack in agents['agent1'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], attack)
    assert attack_status
    assert attacked_agents == [agents['agent3'], agents['agent0']]
    assert not agents['agent0'].active
    assert not agents['agent3'].active
    assert agents['agent1'].active
    assert agents['agent2'].active
    assert agents['agent0'].health <= 0
    assert agents['agent3'].health <= 0
    assert not grid[4, 4]
    assert not grid[3, 2]
    assert grid[2, 2]
    assert grid[2, 3]


    # Attacking nowhere everywhere
    position_state.reset()
    health_state.reset()
    attack = {'attack': np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent1'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent1'], attack)
    assert not attack_status
    assert not attacked_agents
    assert agents['agent0'].active
    assert agents['agent1'].active
    assert agents['agent2'].active
    assert agents['agent3'].active
    assert grid[4, 4]
    assert grid[3, 2]
    assert grid[2, 2]
    assert grid[2, 3]


def test_selective_attack_actor_attack_count():
    grid = Grid(2, 2, overlapping={1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]})
    agents = {
        'agent0': AttackingAgent(
            id='agent0',
            initial_position=np.array([1, 1]),
            encoding=3,
            attack_range=1,
            attack_strength=0,
            attack_accuracy=1,
            attack_count=3
        ),
        'agent1': HealthAgent(
            id='agent1', initial_position=np.array([0, 0]), encoding=1, initial_health=1
        ),
        'agent2': HealthAgent(
            id='agent2', initial_position=np.array([1, 1]), encoding=2, initial_health=1
        ),
        'agent3': HealthAgent(
            id='agent3', initial_position=np.array([1, 1]), encoding=2, initial_health=1
        ),
        'agent4': HealthAgent(
            id='agent4', initial_position=np.array([1, 1]), encoding=1, initial_health=1
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = SelectiveAttackActor(attack_mapping={3: [1, 2]}, grid=grid, agents=agents)
    assert isinstance(attack_actor, ActorBaseComponent)
    assert attack_actor.key == 'attack'
    assert attack_actor.supported_agent_type == AttackingAgent
    assert agents['agent0'].action_space['attack'] == Box(0, 3, (3, 3), int)
    agents['agent0'].finalize()
    np.testing.assert_array_equal(
        agents['agent0'].null_action['attack'],
        np.zeros((3, 3), dtype=int)
    )

    position_state.reset()
    health_state.reset()

    attack = {'attack': np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent0'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], attack)
    assert not attack_status
    assert len(attacked_agents) == 0

    attack = {'attack': np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent0'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], attack)
    assert attack_status
    assert len(attacked_agents) == 2
    assert agents['agent1'] in attacked_agents

    attack = {'attack': np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent0'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], attack)
    assert attack_status
    assert len(attacked_agents) == 3
    assert agents['agent1'] in attacked_agents

    attack = {'attack': np.array([
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent0'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], attack)
    assert attack_status
    assert len(attacked_agents) == 4

    agents['agent0'].attack_strength = 1
    attack = {'attack': np.array([
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent0'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], attack)
    assert attack_status
    assert len(attacked_agents) == 4
    assert not agents['agent1'].active
    assert not agents['agent2'].active
    assert not agents['agent3'].active
    assert not agents['agent4'].active


def test_selective_attack_actor_stacked_attack():
    np.random.seed(24)
    grid = Grid(2, 2, overlapping={1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]})
    agents = {
        'agent0': AttackingAgent(
            id='agent0',
            initial_position=np.array([1, 1]),
            encoding=3,
            attack_range=1,
            attack_strength=1,
            attack_accuracy=1,
            attack_count=3
        ),
        'agent1': HealthAgent(
            id='agent1', initial_position=np.array([0, 0]), encoding=1, initial_health=1
        ),
        'agent2': HealthAgent(
            id='agent2', initial_position=np.array([1, 1]), encoding=2, initial_health=1
        ),
        'agent3': HealthAgent(
            id='agent3', initial_position=np.array([1, 1]), encoding=2, initial_health=1
        ),
        'agent4': HealthAgent(
            id='agent4', initial_position=np.array([1, 1]), encoding=1, initial_health=1
        ),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = SelectiveAttackActor(
        attack_mapping={3: [1, 2]}, grid=grid, agents=agents
    )
    agents['agent0'].finalize()

    position_state.reset()
    health_state.reset()

    attack = {'attack': np.array([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent0'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], attack)
    assert attack_status
    assert len(attacked_agents) == 2
    assert agents['agent2'] in attacked_agents
    assert agents['agent4'] in attacked_agents
    assert not agents['agent2'].active
    assert not agents['agent4'].active

    agents['agent0'].attack_strength = 0.5
    attack_actor = SelectiveAttackActor(
        attack_mapping={3: [1, 2]}, stacked_attacks=True, grid=grid, agents=agents
    )

    attack = {'attack': np.array([
        [1, 0, 0],
        [0, 3, 0],
        [0, 0, 0]
    ], dtype=int)}
    assert attack in agents['agent0'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], attack)
    assert attack_status
    assert len(attacked_agents) == 4
    assert attacked_agents[0] == agents['agent1']
    assert attacked_agents[1] == agents['agent3']
    assert attacked_agents[2] == agents['agent3']
    assert attacked_agents[3] == agents['agent3']
    assert agents['agent1'].active
    assert not agents['agent3'].active

    attack = {'attack': np.array([
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3]
    ], dtype=int)}
    assert attack in agents['agent0'].action_space
    attack_status, attacked_agents = attack_actor.process_action(agents['agent0'], attack)
    assert attack_status
    assert len(attacked_agents) == 3
    assert attacked_agents[0] == agents['agent1']
    assert attacked_agents[1] == agents['agent1']
    assert attacked_agents[2] == agents['agent1']
    assert not agents['agent1'].active


def test_encoding_based_attack_actor():
    grid = Grid(2, 2, overlapping={1: [3], 3: [1]})
    agents = {
        'agent0': HealthAgent(id='agent0', initial_position=np.array([0, 0]), encoding=1),
        'agent1': HealthAgent(id='agent1', initial_position=np.array([0, 1]), encoding=2),
        'agent2': HealthAgent(id='agent2', initial_position=np.array([1, 0]), encoding=2),
        'agent3': AttackingAgent(
            id='agent3',
            initial_position=np.array([1, 1]),
            encoding=3,
            attack_range=1,
            attack_strength=0,
            attack_accuracy=1
        ),
        'agent4': HealthAgent(id='agent4', initial_position=np.array([1, 1]), encoding=1),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = EncodingBasedAttackActor(attack_mapping={3: [1, 2]}, grid=grid, agents=agents)
    assert isinstance(attack_actor, ActorBaseComponent)
    assert attack_actor.key == 'attack'
    assert attack_actor.supported_agent_type == AttackingAgent
    assert agents['agent3'].action_space['attack'] == Dict({
        1: Discrete(2),
        2: Discrete(2)
    })

    agents['agent3'].finalize()
    assert agents['agent3'].null_action == {'attack': {1: 0, 2: 0}}

    position_state.reset()
    health_state.reset()
    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1:0, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 1
    assert attacked_agents[0].encoding == 2
    assert attacked_agents[0].active # Should still be active because attacking agent is weak.

    agents['agent3'].attack_strength = 1
    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 1, 2: 0}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 1
    assert attacked_agents[0].encoding == 1
    assert not attacked_agents[0].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 1, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].encoding == 2
    assert not attacked_agents[0].active
    assert not attacked_agents[1].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 1, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 1
    assert attacked_agents[0].encoding == 2
    assert not attacked_agents[0].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 1, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 0


def test_encoding_based_attack_actor_attack_count():
    grid = Grid(2, 2, overlapping={1: [3], 3: [1]})
    agents = {
        'agent0': HealthAgent(id='agent0', initial_position=np.array([0, 0]), encoding=1),
        'agent1': HealthAgent(id='agent1', initial_position=np.array([0, 1]), encoding=2),
        'agent2': HealthAgent(id='agent2', initial_position=np.array([1, 0]), encoding=2),
        'agent3': AttackingAgent(
            id='agent3',
            initial_position=np.array([1, 1]),
            encoding=3,
            attack_range=1,
            attack_strength=0,
            attack_accuracy=1,
            attack_count=2
        ),
        'agent4': HealthAgent(id='agent4', initial_position=np.array([1, 1]), encoding=1),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = EncodingBasedAttackActor(attack_mapping={3: [1, 2]}, grid=grid, agents=agents)
    assert agents['agent3'].action_space['attack'] == Dict({
        1: Discrete(3),
        2: Discrete(3)
    })

    agents['agent3'].finalize()
    assert agents['agent3'].null_action == {'attack': {1: 0, 2: 0}}

    position_state.reset()
    health_state.reset()
    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1:0, 2: 0}})
    assert not attack_status
    assert type(attacked_agents) is list
    assert not attacked_agents

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1:1, 2: 0}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 1
    assert attacked_agents[0].encoding == 1

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1:0, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 1
    assert attacked_agents[0].encoding == 2

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1:1, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].encoding == 2

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1:2, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 3
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].encoding == 1
    assert attacked_agents[2].encoding == 2

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1:1, 2: 2}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 3
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].encoding == 2
    assert attacked_agents[2].encoding == 2

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1:2, 2: 2}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 4
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].encoding == 1
    assert attacked_agents[2].encoding == 2
    assert attacked_agents[3].encoding == 2

    agents['agent3'].attack_strength = 1
    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 2, 2: 0}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].encoding == 1
    assert not attacked_agents[0].active
    assert not attacked_agents[1].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 2, 2: 2}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0].encoding == 2
    assert attacked_agents[1].encoding == 2
    assert not attacked_agents[0].active
    assert not attacked_agents[1].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 1, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 0


def test_encoding_based_attack_actor_stacked_attack():
    grid = Grid(2, 2, overlapping={1: [3], 3: [1]})
    agents = {
        'agent0': HealthAgent(id='agent0', initial_position=np.array([0, 0]), encoding=1),
        'agent1': HealthAgent(id='agent1', initial_position=np.array([0, 1]), encoding=2),
        'agent2': HealthAgent(id='agent2', initial_position=np.array([1, 0]), encoding=2),
        'agent3': AttackingAgent(
            id='agent3',
            initial_position=np.array([1, 1]),
            encoding=3,
            attack_range=1,
            attack_strength=1,
            attack_accuracy=1,
            attack_count=2
        ),
        'agent4': HealthAgent(id='agent4', initial_position=np.array([1, 1]), encoding=1),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = EncodingBasedAttackActor(
        attack_mapping={3: [1, 2]}, stacked_attacks=True, grid=grid, agents=agents
    )
    assert isinstance(attack_actor, ActorBaseComponent)
    assert attack_actor.key == 'attack'
    assert attack_actor.supported_agent_type == AttackingAgent
    assert agents['agent3'].action_space['attack'] == Dict({
        1: Discrete(3),
        2: Discrete(3)
    })

    agents['agent3'].finalize()
    assert agents['agent3'].null_action == {'attack': {1: 0, 2: 0}}

    position_state.reset()
    health_state.reset()

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 1, 2: 1}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].encoding == 2
    assert not attacked_agents[0].active
    assert not attacked_agents[1].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 2, 2: 0}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0] == attacked_agents[1]
    assert attacked_agents[0].encoding == 1
    assert not attacked_agents[0].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': {1: 2, 2: 2}})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0] == attacked_agents[1]
    assert attacked_agents[0].encoding == 2
    assert not attacked_agents[0].active


def test_restricted_selective_attack_actor():
    grid = Grid(2, 2, overlapping={1: [1]})
    agents = {
        'agent0': HealthAgent(id='agent0', initial_position=np.array([0, 0]), encoding=1),
        'agent1': HealthAgent(id='agent1', initial_position=np.array([0, 1]), encoding=2),
        'agent2': HealthAgent(id='agent2', initial_position=np.array([1, 0]), encoding=2),
        'agent3': AttackingAgent(
            id='agent3',
            initial_position=np.array([1, 1]),
            encoding=3,
            attack_range=1,
            attack_strength=0,
            attack_accuracy=1,
            attack_count=2
        ),
        'agent4': HealthAgent(id='agent4', initial_position=np.array([0, 0]), encoding=1),
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = RestrictedSelectiveAttackActor(
        attack_mapping={3: [1, 2]}, grid=grid, agents=agents
    )
    assert isinstance(attack_actor, ActorBaseComponent)
    assert attack_actor.key == 'attack'
    assert attack_actor.supported_agent_type == AttackingAgent
    assert agents['agent3'].action_space['attack'] == MultiDiscrete([10, 10])

    agents['agent3'].finalize()
    np.testing.assert_array_equal(
        agents['agent3'].null_action['attack'], np.zeros((2,), dtype=int)
    )

    position_state.reset()
    health_state.reset()
    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': [0, 0]})
    assert not attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 0

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': [1, 1]})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0].id != attacked_agents[1].id
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].encoding == 1
    assert attacked_agents[0].active
    assert attacked_agents[1].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': [2, 2]})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 1
    assert attacked_agents[0].active
    assert attacked_agents[0].encoding == 2

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': [1, 4]})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0].active
    assert attacked_agents[0].encoding == 1
    assert attacked_agents[1].active
    assert attacked_agents[1].encoding == 2


def test_restricted_selective_attack_actor_stacked_attacks():
    grid = Grid(2, 2, overlapping={1: [1]})
    agents = {
        'agent0': HealthAgent(
            id='agent0', initial_position=np.array([0, 0]), encoding=1, initial_health=1
        ),
        'agent1': HealthAgent(
            id='agent1', initial_position=np.array([0, 1]), encoding=2, initial_health=1
        ),
        'agent2': HealthAgent(
            id='agent2', initial_position=np.array([1, 0]), encoding=2, initial_health=1
        ),
        'agent3': AttackingAgent(
            id='agent3',
            initial_position=np.array([1, 1]),
            encoding=3,
            attack_range=1,
            attack_strength=1,
            attack_accuracy=1,
            attack_count=2
        )
    }

    position_state = PositionState(grid=grid, agents=agents)
    health_state = HealthState(grid=grid, agents=agents)
    attack_actor = RestrictedSelectiveAttackActor(
        attack_mapping={3: [1, 2]}, stacked_attacks=True, grid=grid, agents=agents
    )
    assert isinstance(attack_actor, ActorBaseComponent)
    assert agents['agent3'].action_space['attack'] == MultiDiscrete([10, 10])

    agents['agent3'].finalize()
    np.testing.assert_array_equal(
        agents['agent3'].null_action['attack'], np.zeros((2,), dtype=int)
    )

    position_state.reset()
    health_state.reset()
    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': [0, 0]})
    assert not attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 0

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': [1, 1]})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0] == attacked_agents[1]
    assert attacked_agents[0].encoding == 1
    assert not attacked_agents[0].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': [2, 2]})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0] == attacked_agents[1]
    assert attacked_agents[0].encoding == 2
    assert not attacked_agents[0].active

    attack_status, attacked_agents = \
        attack_actor.process_action(agents['agent3'], {'attack': [4, 4]})
    assert attack_status
    assert type(attacked_agents) is list
    assert len(attacked_agents) == 2
    assert attacked_agents[0] == attacked_agents[1]
    assert attacked_agents[0].encoding == 2
    assert not attacked_agents[0].active
