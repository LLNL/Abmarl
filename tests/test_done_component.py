
from admiral.component_envs.done_component import LifeAgent, TeamAgent
from admiral.component_envs.done_component import DeadDoneComponent, TeamDeadDoneComponent

class DoneTestAgent(LifeAgent, TeamAgent): pass

def test_dead_done_condition():
    agents = {
        'agent0': LifeAgent(id='agent0'),
        'agent1': LifeAgent(id='agent1'),
        'agent2': LifeAgent(id='agent2'),
        'agent3': LifeAgent(id='agent3'),
    }
    component = DeadDoneComponent(agents=agents)
    assert component.get_done('agent0') is False
    assert component.get_done('agent1') is False
    assert component.get_done('agent2') is False
    assert component.get_done('agent3') is False
    assert component.get_all_done() is False

    agents['agent0'].is_alive = False
    agents['agent1'].is_alive = False
    assert component.get_done('agent0')
    assert component.get_done('agent1')
    assert component.get_done('agent2') is False
    assert component.get_done('agent3') is False
    assert component.get_all_done() is False

    agents['agent2'].is_alive = False
    agents['agent3'].is_alive = False
    assert component.get_done('agent0')
    assert component.get_done('agent1')
    assert component.get_done('agent2')
    assert component.get_done('agent3')
    assert component.get_all_done()

def test_team_dead_done_condition():
    agents = {
        'agent0': DoneTestAgent(id='agent0', team=0),
        'agent1': DoneTestAgent(id='agent1', team=1),
        'agent2': DoneTestAgent(id='agent2', team=0),
        'agent3': DoneTestAgent(id='agent3', team=1),
        'agent4': DoneTestAgent(id='agent4', team=0),
        'agent5': DoneTestAgent(id='agent5', team=2),
    }
    component = TeamDeadDoneComponent(agents=agents, number_of_teams=3)

    assert component.get_done('agent0') is False
    assert component.get_done('agent1') is False
    assert component.get_done('agent2') is False
    assert component.get_done('agent3') is False
    assert component.get_done('agent4') is False
    assert component.get_done('agent5') is False
    assert component.get_all_done() is False

    agents['agent5'].is_alive = False
    agents['agent4'].is_alive = False
    assert component.get_done('agent0') is False
    assert component.get_done('agent1') is False
    assert component.get_done('agent2') is False
    assert component.get_done('agent3') is False
    assert component.get_done('agent4')
    assert component.get_done('agent5')
    assert component.get_all_done() is False

    agents['agent1'].is_alive = False
    agents['agent3'].is_alive = False
    assert component.get_done('agent0') is False
    assert component.get_done('agent1')
    assert component.get_done('agent2') is False
    assert component.get_done('agent3')
    assert component.get_done('agent4')
    assert component.get_done('agent5')
    assert component.get_all_done()
