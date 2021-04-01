
from gym.spaces import MultiBinary
import numpy as np

from admiral.envs.components.agent import AttackingAgent, PositionAgent, TeamAgent, LifeAgent
from admiral.envs.components.actor import AttackActor, PositionTeamBasedAttackActor


class AttackTestAgent(AttackingAgent, PositionAgent, LifeAgent, TeamAgent):
    pass

def test_position_based_attack_actor():
    agents = {
        'agent0': AttackTestAgent(id='agent0', attack_range=1, initial_position=np.array([1, 1]), \
            attack_strength=0.6),
        'agent1': AttackTestAgent(id='agent1', attack_range=1, initial_position=np.array([0, 1]), \
            attack_strength=0.6),
        'agent2': AttackTestAgent(id='agent2', attack_range=1, initial_position=np.array([4, 2]), \
            attack_strength=0.6),
        'agent3': AttackTestAgent(id='agent3', attack_range=1, initial_position=np.array([4, 3]), \
            attack_strength=0.6),
        'agent4': AttackTestAgent(id='agent4', attack_range=0, initial_position=np.array([3, 2]), \
            attack_strength=0.6),
        'agent5': AttackTestAgent(id='agent5', attack_range=2, initial_position=np.array([4, 0]), \
            attack_strength=0.6),
    }

    assert agents['agent0'].attack_range == 1
    np.testing.assert_array_equal(agents['agent0'].initial_position, np.array([1, 1]))
    assert agents['agent1'].attack_range == 1
    np.testing.assert_array_equal(agents['agent1'].initial_position, np.array([0, 1]))
    assert agents['agent2'].attack_range == 1
    np.testing.assert_array_equal(agents['agent2'].initial_position, np.array([4, 2]))
    assert agents['agent3'].attack_range == 1
    np.testing.assert_array_equal(agents['agent3'].initial_position, np.array([4, 3]))
    assert agents['agent4'].attack_range == 0
    np.testing.assert_array_equal(agents['agent4'].initial_position, np.array([3, 2]))
    assert agents['agent5'].attack_range == 2
    np.testing.assert_array_equal(agents['agent5'].initial_position, np.array([4, 0]))

    actor = AttackActor(agents=agents)
    for agent in agents.values():
        agent.position = agent.initial_position

    assert actor.process_attack(agents['agent0'], True).id == 'agent1'
    assert actor.process_attack(agents['agent1'], True).id == 'agent0'
    assert actor.process_attack(agents['agent2'], True).id == 'agent3'
    assert actor.process_attack(agents['agent3'], True).id == 'agent2'
    assert actor.process_attack(agents['agent4'], True) is None
    assert actor.process_attack(agents['agent5'], True).id == 'agent2'

    agents['agent0'].is_alive = False
    agents['agent2'].is_alive = False

    assert actor.process_attack(agents['agent1'], True) is None
    assert actor.process_attack(agents['agent3'], True).id == 'agent4'
    assert actor.process_attack(agents['agent4'], True) is None
    assert actor.process_attack(agents['agent5'], True).id == 'agent4'

def test_position_team_based_attack_actor():
    agents = {
        'agent0': AttackTestAgent(id='agent0', attack_range=1, initial_position=np.array([1, 1]), \
            attack_strength=0.6, team=1),
        'agent1': AttackTestAgent(id='agent1', attack_range=1, initial_position=np.array([0, 1]), \
            attack_strength=0.6, team=2),
        'agent2': AttackTestAgent(id='agent2', attack_range=1, initial_position=np.array([4, 2]), \
            attack_strength=0.6, team=1),
        'agent3': AttackTestAgent(id='agent3', attack_range=1, initial_position=np.array([4, 3]), \
            attack_strength=0.6, team=1),
        'agent4': AttackTestAgent(id='agent4', attack_range=0, initial_position=np.array([3, 2]), \
            attack_strength=0.6, team=3),
        'agent5': AttackTestAgent(id='agent5', attack_range=2, initial_position=np.array([4, 0]), \
            attack_strength=0.6, team=1),
    }

    assert agents['agent0'].attack_range == 1
    assert agents['agent0'].team == 1
    np.testing.assert_array_equal(agents['agent0'].initial_position, np.array([1, 1]))
    assert agents['agent1'].attack_range == 1
    assert agents['agent1'].team == 2
    np.testing.assert_array_equal(agents['agent1'].initial_position, np.array([0, 1]))
    assert agents['agent2'].attack_range == 1
    assert agents['agent2'].team == 1
    np.testing.assert_array_equal(agents['agent2'].initial_position, np.array([4, 2]))
    assert agents['agent3'].attack_range == 1
    assert agents['agent3'].team == 1
    np.testing.assert_array_equal(agents['agent3'].initial_position, np.array([4, 3]))
    assert agents['agent4'].attack_range == 0
    assert agents['agent4'].team == 3
    np.testing.assert_array_equal(agents['agent4'].initial_position, np.array([3, 2]))
    assert agents['agent5'].attack_range == 2
    assert agents['agent5'].team == 1
    np.testing.assert_array_equal(agents['agent5'].initial_position, np.array([4, 0]))

    for agent in agents.values():
        agent.position = agent.initial_position
    actor = PositionTeamBasedAttackActor(agents=agents)

    assert actor.process_attack(agents['agent0'], True).id == 'agent1'
    assert actor.process_attack(agents['agent1'], True).id == 'agent0'
    assert actor.process_attack(agents['agent2'], True).id == 'agent4'
    assert actor.process_attack(agents['agent3'], True).id == 'agent4'
    assert actor.process_attack(agents['agent4'], True) is None
    assert actor.process_attack(agents['agent5'], True).id == 'agent4'

    agents['agent4'].is_alive = False
    agents['agent0'].is_alive = False

    assert actor.process_attack(agents['agent1'], True) is None
    assert actor.process_attack(agents['agent2'], True) is None
    assert actor.process_attack(agents['agent3'], True) is None
    assert actor.process_attack(agents['agent5'], True) is None

def test_attack_accuracy():
    np.random.seed(24)
    agents = {
        'agent0': AttackTestAgent(id='agent0', attack_range=1, attack_strength=0, attack_accuracy=0, initial_position=np.array([1,1])),
        'agent1': AttackTestAgent(id='agent1', attack_range=0, attack_strength=0, initial_position=np.array([0,0]))
    }

    assert agents['agent0'].attack_accuracy == 0
    assert agents['agent1'].attack_accuracy == 1

    for agent in agents.values():
        agent.position = agent.initial_position

    actor = AttackActor(agents=agents)
    assert actor.process_attack(agents['agent0'], True) is None # Action failed because low accuracy

    agents['agent0'].attack_accuracy = 0.5
    assert actor.process_attack(agents['agent0'], True) is None
    assert actor.process_attack(agents['agent0'], True) is None
    assert actor.process_attack(agents['agent0'], True).id == 'agent1'
    assert actor.process_attack(agents['agent0'], True).id == 'agent1'
    assert actor.process_attack(agents['agent0'], True) is None
    assert actor.process_attack(agents['agent0'], True) is None
