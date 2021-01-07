
from admiral.envs.components.agent import LifeAgent, TeamAgent
from admiral.envs.components.state import LifeState
from admiral.envs.components.done import DeadDone, TeamDeadDone

class DoneTestAgent(LifeAgent, TeamAgent): pass

def test_dead_done_condition():
    agents = {
        'agent0': LifeAgent(id='agent0'),
        'agent1': LifeAgent(id='agent1'),
        'agent2': LifeAgent(id='agent2'),
        'agent3': LifeAgent(id='agent3'),
    }
    state = LifeState(agents=agents)
    done = DeadDone(agents=agents)
    state.reset()

    assert done.get_done(agents['agent0']) is False
    assert done.get_done(agents['agent1']) is False
    assert done.get_done(agents['agent2']) is False
    assert done.get_done(agents['agent3']) is False
    assert done.get_all_done() is False

    agents['agent0'].is_alive = False
    agents['agent1'].is_alive = False
    assert done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2']) is False
    assert done.get_done(agents['agent3']) is False
    assert done.get_all_done() is False

    agents['agent2'].is_alive = False
    agents['agent3'].is_alive = False
    assert done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2'])
    assert done.get_done(agents['agent3'])
    assert done.get_all_done()

def test_team_dead_done_condition():
    agents = {
        'agent0': DoneTestAgent(id='agent0', team=0),
        'agent1': DoneTestAgent(id='agent1', team=1),
        'agent2': DoneTestAgent(id='agent2', team=0),
        'agent3': DoneTestAgent(id='agent3', team=1),
        'agent4': DoneTestAgent(id='agent4', team=0),
        'agent5': DoneTestAgent(id='agent5', team=2),
    }
    state = LifeState(agents=agents)
    done = TeamDeadDone(agents=agents, number_of_teams=3)
    state.reset()

    assert not done.get_done(agents['agent0'])
    assert not done.get_done(agents['agent1'])
    assert not done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert not done.get_done(agents['agent4'])
    assert not done.get_done(agents['agent5'])
    assert not done.get_all_done()

    agents['agent5'].is_alive = False
    agents['agent4'].is_alive = False
    assert not done.get_done(agents['agent0'])
    assert not done.get_done(agents['agent1'])
    assert not done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert     done.get_done(agents['agent4'])
    assert     done.get_done(agents['agent5'])
    assert not done.get_all_done()

    agents['agent1'].is_alive = False
    agents['agent3'].is_alive = False
    assert not done.get_done(agents['agent0'])
    assert     done.get_done(agents['agent1'])
    assert not done.get_done(agents['agent2'])
    assert     done.get_done(agents['agent3'])
    assert     done.get_done(agents['agent4'])
    assert     done.get_done(agents['agent5'])
    assert     done.get_all_done()
