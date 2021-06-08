import numpy as np

from abmarl.sim.components.agent import AttackingAgent
from abmarl.sim.components.actor import AttackActor


def test_position_based_attack_actor():
    agents = {
        'agent0': AttackingAgent(
            id='agent0', attack_range=1, initial_position=np.array([1, 1]), attack_strength=0.6
        ),
        'agent1': AttackingAgent(
            id='agent1', attack_range=1, initial_position=np.array([0, 1]), attack_strength=0.6
        ),
        'agent2': AttackingAgent(
            id='agent2', attack_range=1, initial_position=np.array([4, 2]), attack_strength=0.6
        ),
        'agent3': AttackingAgent(
            id='agent3', attack_range=1, initial_position=np.array([4, 3]), attack_strength=0.6
        ),
        'agent4': AttackingAgent(
            id='agent4', attack_range=0, initial_position=np.array([3, 2]), attack_strength=0.6
        ),
        'agent5': AttackingAgent(
            id='agent5', attack_range=2, initial_position=np.array([4, 0]), attack_strength=0.6
        ),
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

    assert actor.process_action(agents['agent0'], {'attack': True}).id == 'agent1'
    assert actor.process_action(agents['agent1'], {'attack': True}).id == 'agent0'
    assert actor.process_action(agents['agent2'], {'attack': True}).id == 'agent3'
    assert actor.process_action(agents['agent3'], {'attack': True}).id == 'agent2'
    assert actor.process_action(agents['agent4'], {'attack': True}) is None
    assert actor.process_action(agents['agent5'], {'attack': True}).id == 'agent2'

    agents['agent0'].is_alive = False
    agents['agent2'].is_alive = False

    assert actor.process_action(agents['agent1'], {'attack': True}) is None
    assert actor.process_action(agents['agent3'], {'attack': True}).id == 'agent4'
    assert actor.process_action(agents['agent4'], {'attack': True}) is None
    assert actor.process_action(agents['agent5'], {'attack': True}).id == 'agent4'


def test_position_team_based_attack_actor():
    agents = {
        'agent0': AttackingAgent(
            id='agent0', attack_range=1, initial_position=np.array([1, 1]), attack_strength=0.6,
            team=1
        ),
        'agent1': AttackingAgent(
            id='agent1', attack_range=1, initial_position=np.array([0, 1]), attack_strength=0.6,
            team=2
        ),
        'agent2': AttackingAgent(
            id='agent2', attack_range=1, initial_position=np.array([4, 2]), attack_strength=0.6,
            team=1
        ),
        'agent3': AttackingAgent(
            id='agent3', attack_range=1, initial_position=np.array([4, 3]), attack_strength=0.6,
            team=1
        ),
        'agent4': AttackingAgent(
            id='agent4', attack_range=0, initial_position=np.array([3, 2]), attack_strength=0.6,
            team=3
        ),
        'agent5': AttackingAgent(
            id='agent5', attack_range=2, initial_position=np.array([4, 0]), attack_strength=0.6,
            team=1
        ),
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
    actor = AttackActor(agents=agents, number_of_teams=3)

    assert actor.process_action(agents['agent0'], {'attack': True}).id == 'agent1'
    assert actor.process_action(agents['agent1'], {'attack': True}).id == 'agent0'
    assert actor.process_action(agents['agent2'], {'attack': True}).id == 'agent4'
    assert actor.process_action(agents['agent3'], {'attack': True}).id == 'agent4'
    assert actor.process_action(agents['agent4'], {'attack': True}) is None
    assert actor.process_action(agents['agent5'], {'attack': True}).id == 'agent4'

    agents['agent4'].is_alive = False
    agents['agent0'].is_alive = False

    assert actor.process_action(agents['agent1'], {'attack': True}) is None
    assert actor.process_action(agents['agent2'], {'attack': True}) is None
    assert actor.process_action(agents['agent3'], {'attack': True}) is None
    assert actor.process_action(agents['agent5'], {'attack': True}) is None


def test_attack_accuracy():
    np.random.seed(24)
    agents = {
        'agent0': AttackingAgent(
            id='agent0', attack_range=1, attack_strength=0, attack_accuracy=0,
            initial_position=np.array([1,1])
        ),
        'agent1': AttackingAgent(
            id='agent1', attack_range=0, attack_strength=0, initial_position=np.array([0,0])
        )
    }

    assert agents['agent0'].attack_accuracy == 0
    assert agents['agent1'].attack_accuracy == 1

    for agent in agents.values():
        agent.position = agent.initial_position

    actor = AttackActor(agents=agents)
    # Action failed because low accuracy
    assert actor.process_action(agents['agent0'], {'attack': True}) is None

    agents['agent0'].attack_accuracy = 0.5
    assert actor.process_action(agents['agent0'], {'attack': True}) is None
    assert actor.process_action(agents['agent0'], {'attack': True}) is None
    assert actor.process_action(agents['agent0'], {'attack': True}).id == 'agent1'
    assert actor.process_action(agents['agent0'], {'attack': True}).id == 'agent1'
    assert actor.process_action(agents['agent0'], {'attack': True}) is None
    assert actor.process_action(agents['agent0'], {'attack': True}) is None


def test_team_matrix():
    agents = {
        'agent0': AttackingAgent(
            id='agent0', attack_range=1, initial_position=np.array([1, 1]), attack_strength=0.6,
            team=1
        ),
        'agent1': AttackingAgent(
            id='agent1', attack_range=4, initial_position=np.array([0, 1]), attack_strength=0.6,
            team=2
        ),
        'agent2': AttackingAgent(
            id='agent2', attack_range=1, initial_position=np.array([4, 2]), attack_strength=0.6,
            team=1
        ),
        'agent3': AttackingAgent(
            id='agent3', attack_range=1, initial_position=np.array([4, 3]), attack_strength=0.6
        ),
        'agent4': AttackingAgent(
            id='agent4', attack_range=1, initial_position=np.array([3, 2]), attack_strength=0.6,
            team=3
        ),
        'agent5': AttackingAgent(
            id='agent5', attack_range=1, initial_position=np.array([4, 0]), attack_strength=0.6,
            team=1
        ),
    }

    for agent in agents.values():
        agent.position = agent.initial_position

    team_attack_matrix = np.zeros((4,4))
    team_attack_matrix[0, :] = 1
    team_attack_matrix[1, 0] = 1
    team_attack_matrix[2, 3] = 1

    actor = AttackActor(agents=agents, number_of_teams=3, team_attack_matrix=team_attack_matrix)
    assert actor.process_action(agents['agent0'], {'attack': True}) is None
    assert actor.process_action(agents['agent1'], {'attack': True}).id == 'agent4'
    assert actor.process_action(agents['agent2'], {'attack': True}).id == 'agent3'
    assert actor.process_action(agents['agent3'], {'attack': True}).id == 'agent2'
    assert actor.process_action(agents['agent4'], {'attack': True}) is None
    assert actor.process_action(agents['agent5'], {'attack': True}) is None
