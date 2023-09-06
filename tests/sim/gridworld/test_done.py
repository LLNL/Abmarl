
import numpy as np

from abmarl.sim.gridworld.agent import HealthAgent, MovingAgent, GridWorldAgent
from abmarl.sim.gridworld.state import HealthState, PositionState
from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.done import ActiveDone, TargetAgentDone, TargetDestroyedDone, \
    DoneBaseComponent
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


def test_target_overlap_done():
    grid = Grid(2, 2, overlapping={1: {1, 2}, 2: {1, 2}})
    agents = {
        'agent0': MovingAgent(
            id='agent0', encoding=1, move_range=1, initial_position=np.array([0, 0])
        ),
        'agent1': MovingAgent(
            id='agent1', encoding=1, move_range=1, initial_position=np.array([0, 1])
        ),
        'agent2': MovingAgent(
            id='agent2', encoding=2, move_range=1, initial_position=np.array([1, 0])
        ),
        'agent3': MovingAgent(
            id='agent3', encoding=2, move_range=1, initial_position=np.array([1, 1])
        )
    }
    target_mapping = {
        'agent0': 'agent1',
        'agent1': 'agent2',
        'agent2': 'agent3',
        'agent3': 'agent0'
    }
    state = PositionState(grid=grid, agents=agents)
    actor = MoveActor(grid=grid, agents=agents)
    target_done = TargetAgentDone(grid=grid, agents=agents, target_mapping=target_mapping)
    state.reset()
    for agent in agents.values():
        agent.finalize()

    assert not target_done.get_done(agents['agent0'])
    assert not target_done.get_done(agents['agent1'])
    assert not target_done.get_done(agents['agent2'])
    assert not target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    actor.process_action(agents['agent0'], {'move': np.array([0, 1])})
    np.testing.assert_array_equal(agents['agent0'].position, agents['agent1'].position)
    assert target_done.get_done(agents['agent0'])
    assert not target_done.get_done(agents['agent1'])
    assert not target_done.get_done(agents['agent2'])
    assert not target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    actor.process_action(agents['agent1'], {'move': np.array([1, -1])})
    np.testing.assert_array_equal(agents['agent1'].position, agents['agent2'].position)
    assert not target_done.get_done(agents['agent0'])
    assert target_done.get_done(agents['agent1'])
    assert not target_done.get_done(agents['agent2'])
    assert not target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    actor.process_action(agents['agent3'], {'move': np.array([-1, 0])})
    np.testing.assert_array_equal(agents['agent0'].position, agents['agent3'].position)
    assert not target_done.get_done(agents['agent0'])
    assert target_done.get_done(agents['agent1'])
    assert not target_done.get_done(agents['agent2'])
    assert target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    actor.process_action(agents['agent2'], {'move': np.array([-1, 1])})
    np.testing.assert_array_equal(agents['agent2'].position, agents['agent3'].position)
    assert not target_done.get_done(agents['agent0'])
    assert not target_done.get_done(agents['agent1'])
    assert target_done.get_done(agents['agent2'])
    assert target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    actor.process_action(agents['agent1'], {'move': np.array([-1, 1])})
    assert target_done.get_done(agents['agent0'])
    assert target_done.get_done(agents['agent1'])
    assert target_done.get_done(agents['agent2'])
    assert target_done.get_done(agents['agent3'])
    assert target_done.get_all_done()


def test_target_destroyed_done():
    grid = Grid(2, 2, overlapping={1: {1, 2}, 2: {1, 2}})
    agents = {
        'agent0': GridWorldAgent(
            id='agent0', encoding=1, initial_position=np.array([0, 0])
        ),
        'agent1': GridWorldAgent(
            id='agent1', encoding=1, initial_position=np.array([0, 1])
        ),
        'agent2': GridWorldAgent(
            id='agent2', encoding=2, initial_position=np.array([1, 0])
        ),
        'agent3': GridWorldAgent(
            id='agent3', encoding=2, initial_position=np.array([1, 1])
        )
    }
    target_mapping = {
        'agent0': 'agent1',
        'agent1': 'agent2',
        'agent2': 'agent3',
        'agent3': 'agent0'
    }
    state = PositionState(grid=grid, agents=agents)
    target_done = TargetDestroyedDone(grid=grid, agents=agents, target_mapping=target_mapping)
    state.reset()

    assert not target_done.get_done(agents['agent0'])
    assert not target_done.get_done(agents['agent1'])
    assert not target_done.get_done(agents['agent2'])
    assert not target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    agents['agent1'].active = False
    assert target_done.get_done(agents['agent0'])
    assert not target_done.get_done(agents['agent1'])
    assert not target_done.get_done(agents['agent2'])
    assert not target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    agents['agent1'].active = True
    agents['agent2'].active = False
    assert not target_done.get_done(agents['agent0'])
    assert target_done.get_done(agents['agent1'])
    assert not target_done.get_done(agents['agent2'])
    assert not target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    agents['agent0'].active = False
    assert not target_done.get_done(agents['agent0'])
    assert target_done.get_done(agents['agent1'])
    assert not target_done.get_done(agents['agent2'])
    assert target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    agents['agent2'].active = True
    agents['agent3'].active = False
    assert not target_done.get_done(agents['agent0'])
    assert not target_done.get_done(agents['agent1'])
    assert target_done.get_done(agents['agent2'])
    assert target_done.get_done(agents['agent3'])
    assert not target_done.get_all_done()

    agents['agent1'].active = False
    agents['agent2'].active = False
    assert target_done.get_done(agents['agent0'])
    assert target_done.get_done(agents['agent1'])
    assert target_done.get_done(agents['agent2'])
    assert target_done.get_done(agents['agent3'])
    assert target_done.get_all_done()
