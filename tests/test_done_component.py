import numpy as np

from abmarl.sim.components.agent import ComponentAgent as Agent
from abmarl.sim.components.state import LifeState, ContinuousPositionState
from abmarl.sim.components.done import DeadDone, TeamDeadDone, TooCloseDone


def test_dead_done_condition():
    agents = {
        'agent0': Agent(id='agent0'),
        'agent1': Agent(id='agent1'),
        'agent2': Agent(id='agent2'),
        'agent3': Agent(id='agent3'),
    }
    state = LifeState(agents=agents)
    done = DeadDone(agents=agents)
    state.reset()

    assert not done.get_done(agents['agent0'])
    assert not done.get_done(agents['agent1'])
    assert not done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert not done.get_all_done()

    agents['agent0'].is_alive = False
    agents['agent1'].is_alive = False
    assert done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert not done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert not done.get_all_done()

    agents['agent2'].is_alive = False
    agents['agent3'].is_alive = False
    assert done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2'])
    assert done.get_done(agents['agent3'])
    assert done.get_all_done()


def test_team_dead_done_condition():
    agents = {
        'agent0': Agent(id='agent0', team=1),
        'agent1': Agent(id='agent1', team=2),
        'agent2': Agent(id='agent2', team=1),
        'agent3': Agent(id='agent3', team=2),
        'agent4': Agent(id='agent4', team=1),
        'agent5': Agent(id='agent5', team=3),
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
    assert done.get_done(agents['agent4'])
    assert done.get_done(agents['agent5'])
    assert not done.get_all_done()

    agents['agent1'].is_alive = False
    agents['agent3'].is_alive = False
    assert not done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert not done.get_done(agents['agent2'])
    assert done.get_done(agents['agent3'])
    assert done.get_done(agents['agent4'])
    assert done.get_done(agents['agent5'])
    assert done.get_all_done()


def test_too_close_done_with_continuous():
    agents = {
        'agent0': Agent(id='agent0', initial_position=np.array([0.1, 0.1])),
        'agent1': Agent(id='agent1', initial_position=np.array([0.24, 0.5])),
        'agent2': Agent(id='agent2', initial_position=np.array([0.3, 0.5])),
        'agent3': Agent(id='agent3', initial_position=np.array([3.76, 3.5])),
        'agent4': Agent(id='agent4', initial_position=np.array([3.75, 3.6])),
        'agent5': Agent(id='agent5', initial_position=np.array([2.5, 3.0])),
    }

    state = ContinuousPositionState(region=4, agents=agents)
    done = TooCloseDone(position=state, agents=agents, collision_distance=0.25)
    state.reset()

    assert done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2'])
    assert done.get_done(agents['agent3'])
    assert done.get_done(agents['agent4'])
    assert not done.get_done(agents['agent5'])
    assert done.get_all_done()
