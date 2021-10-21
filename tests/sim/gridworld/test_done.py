
from abmarl.sim.gridworld.agent import HealthAgent
from abmarl.sim.gridworld.state import HealthState
from abmarl.sim.gridworld.done import ActiveDone, DoneBaseComponent
from abmarl.sim.gridworld.grid import Grid


def test_active_done():
    grid = Grid(2, 3)
    agents = {
        'agent0': HealthAgent(id='agent0', encoding=1),
        'agent1': HealthAgent(id='agent1', encoding=1),
        'agent2': HealthAgent(id='agent2', encoding=1),
    }

    health_state = HealthState(agents=agents, grid=grid)
    health_state.reset()

    active_done = ActiveDone(agents=agents, grid=grid)
    assert isinstance(active_done, DoneBaseComponent)

    assert agents['agent0'].active
    assert agents['agent1'].active
    assert agents['agent2'].active
    assert not active_done.get_done(agents['agent0'])
    assert not active_done.get_done(agents['agent1'])
    assert not active_done.get_done(agents['agent2'])
    assert not active_done.get_all_done()

    agents['agent0'].health = 0
    agents['agent1'].health = 0
    assert active_done.get_done(agents['agent0'])
    assert active_done.get_done(agents['agent1'])
    assert not active_done.get_done(agents['agent2'])
    assert not active_done.get_all_done()

    agents['agent2'].health = 0
    assert active_done.get_done(agents['agent2'])
    assert active_done.get_all_done()
