
import numpy as np

from admiral.component_envs.attacking import GridAttackingAgent, GridPositionAgent, TeamAgent, DyingAgent
from admiral.component_envs.attacking import GridAttackingComponent, GridAttackingTeamComponent

class AttackTestAgent(GridAttackingAgent, GridPositionAgent, DyingAgent):
    pass

class AttackTeamTestAgent(GridAttackingAgent, GridPositionAgent, DyingAgent, TeamAgent):
    pass

def test_grid_attacking_component():
    agents = {
        'agent0': AttackTestAgent(id='agent0', attack_range=1, starting_position=np.array([1, 1]), \
            attack_strength=0.6),
        'agent1': AttackTestAgent(id='agent1', attack_range=1, starting_position=np.array([0, 1]), \
            attack_strength=0.6),
        'agent2': AttackTestAgent(id='agent2', attack_range=1, starting_position=np.array([4, 2]), \
            attack_strength=0.6),
        'agent3': AttackTestAgent(id='agent3', attack_range=1, starting_position=np.array([4, 3]), \
            attack_strength=0.6),
        'agent4': AttackTestAgent(id='agent4', attack_range=0, starting_position=np.array([3, 2]), \
            attack_strength=0.6),
        'agent5': AttackTestAgent(id='agent5', attack_range=2, starting_position=np.array([4, 0]), \
            attack_strength=0.6),
    }

    assert agents['agent0'].attack_range == 1
    np.testing.assert_array_equal(agents['agent0'].starting_position, np.array([1, 1]))
    assert agents['agent1'].attack_range == 1
    np.testing.assert_array_equal(agents['agent1'].starting_position, np.array([0, 1]))
    assert agents['agent2'].attack_range == 1
    np.testing.assert_array_equal(agents['agent2'].starting_position, np.array([4, 2]))
    assert agents['agent3'].attack_range == 1
    np.testing.assert_array_equal(agents['agent3'].starting_position, np.array([4, 3]))
    assert agents['agent4'].attack_range == 0
    np.testing.assert_array_equal(agents['agent4'].starting_position, np.array([3, 2]))
    assert agents['agent5'].attack_range == 2
    np.testing.assert_array_equal(agents['agent5'].starting_position, np.array([4, 0]))

    for agent in agents.values():
        agent.position = agent.starting_position
    component = GridAttackingComponent(agents=agents)

    assert component.act(agents['agent0']) == 'agent1'
    assert component.act(agents['agent1']) == 'agent0'
    assert component.act(agents['agent2']) == 'agent3'
    assert component.act(agents['agent3']) == 'agent2'
    assert component.act(agents['agent4']) is None
    assert component.act(agents['agent5']) == 'agent2'

    agents['agent0'].is_alive = False
    agents['agent2'].is_alive = False

    assert component.act(agents['agent1']) is None
    assert component.act(agents['agent3']) == 'agent4'
    assert component.act(agents['agent4']) is None
    assert component.act(agents['agent5']) == 'agent4'

def test_grid_team_attacking_component():
    agents = {
        'agent0': AttackTeamTestAgent(id='agent0', attack_range=1, starting_position=np.array([1, 1]), \
            attack_strength=0.6, team=1),
        'agent1': AttackTeamTestAgent(id='agent1', attack_range=1, starting_position=np.array([0, 1]), \
            attack_strength=0.6, team=2),
        'agent2': AttackTeamTestAgent(id='agent2', attack_range=1, starting_position=np.array([4, 2]), \
            attack_strength=0.6, team=1),
        'agent3': AttackTeamTestAgent(id='agent3', attack_range=1, starting_position=np.array([4, 3]), \
            attack_strength=0.6, team=1),
        'agent4': AttackTeamTestAgent(id='agent4', attack_range=0, starting_position=np.array([3, 2]), \
            attack_strength=0.6, team=3),
        'agent5': AttackTeamTestAgent(id='agent5', attack_range=2, starting_position=np.array([4, 0]), \
            attack_strength=0.6, team=1),
    }

    assert agents['agent0'].attack_range == 1
    assert agents['agent0'].team == 1
    np.testing.assert_array_equal(agents['agent0'].starting_position, np.array([1, 1]))
    assert agents['agent1'].attack_range == 1
    assert agents['agent1'].team == 2
    np.testing.assert_array_equal(agents['agent1'].starting_position, np.array([0, 1]))
    assert agents['agent2'].attack_range == 1
    assert agents['agent2'].team == 1
    np.testing.assert_array_equal(agents['agent2'].starting_position, np.array([4, 2]))
    assert agents['agent3'].attack_range == 1
    assert agents['agent3'].team == 1
    np.testing.assert_array_equal(agents['agent3'].starting_position, np.array([4, 3]))
    assert agents['agent4'].attack_range == 0
    assert agents['agent4'].team == 3
    np.testing.assert_array_equal(agents['agent4'].starting_position, np.array([3, 2]))
    assert agents['agent5'].attack_range == 2
    assert agents['agent5'].team == 1
    np.testing.assert_array_equal(agents['agent5'].starting_position, np.array([4, 0]))

    for agent in agents.values():
        agent.position = agent.starting_position
    component = GridAttackingTeamComponent(agents=agents)

    assert component.act(agents['agent0']) == 'agent1'
    assert component.act(agents['agent1']) == 'agent0'
    assert component.act(agents['agent2']) == 'agent4'
    assert component.act(agents['agent3']) == 'agent4'
    assert component.act(agents['agent4']) is None
    assert component.act(agents['agent5']) == 'agent4'

    agents['agent4'].is_alive = False
    agents['agent0'].is_alive = False

    assert component.act(agents['agent1']) is None
    assert component.act(agents['agent2']) is None
    assert component.act(agents['agent3']) is None
    assert component.act(agents['agent5']) is None
