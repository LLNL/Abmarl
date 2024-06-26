
import numpy as np
import pytest

from abmarl.sim.gridworld.agent import MovingAgent, GridWorldAgent
from abmarl.sim.gridworld.state import HealthState, PositionState
from abmarl.sim.gridworld.actor import MoveActor
from abmarl.sim.gridworld.done import ActiveDone, TargetAgentOverlapDone, \
    TargetAgentInactiveDone, TargetEncodingInactiveDone, DoneBaseComponent
from abmarl.sim.gridworld.grid import Grid


def test_active_done():
    grid = Grid(2, 3)
    agents = {
        'agent0': GridWorldAgent(id='agent0', encoding=1),
        'agent1': GridWorldAgent(id='agent1', encoding=1),
        'agent2': GridWorldAgent(id='agent2', encoding=1),
    }

    health_state = HealthState(agents=agents, grid=grid)
    health_state.reset()

    active_done = ActiveDone(agents=agents, grid=grid)
    assert isinstance(active_done, DoneBaseComponent)
    assert active_done._encodings_in_sim == {1}

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
    target_done = TargetAgentOverlapDone(grid=grid, agents=agents, target_mapping=target_mapping)
    assert target_done._encodings_in_sim == {1, 2}
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
    target_done = TargetAgentInactiveDone(grid=grid, agents=agents, target_mapping=target_mapping)
    assert target_done._encodings_in_sim == {1, 2}
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


def test_target_encoding_destroyed_done():
    grid = Grid(3, 9)
    agents = {
        f'agent{n}': GridWorldAgent(id=f'agent{n}', encoding=n % 4 + 1) for n in range(12)
    }
    state = PositionState(grid=grid, agents=agents)
    done = TargetEncodingInactiveDone(
        grid=grid,
        agents=agents,
        target_mapping={2: 1, 3: 1, 4:2}
    )
    assert isinstance(done, DoneBaseComponent)
    assert done.target_mapping == {2: {1}, 3: {1}, 4: {2}}
    assert done.sim_ends_if_one_done
    assert done._encodings_in_sim == {1, 2, 3, 4}

    # Test reset state
    state.reset()
    for agent in agents.values():
        assert not done.get_done(agent)
    assert not done.get_all_done()

    # Test encodings that are targeting 0 are done
    agents['agent0'].active = False
    assert not done.get_done(agents['agent1'])
    assert not done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    agents['agent4'].active = False
    agents['agent8'].active = False
    assert not done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert not done.get_done(agents['agent4'])
    assert done.get_done(agents['agent5'])
    assert done.get_done(agents['agent6'])
    assert not done.get_done(agents['agent7'])
    assert not done.get_done(agents['agent8'])
    assert done.get_done(agents['agent9'])
    assert done.get_done(agents['agent10'])
    assert not done.get_done(agents['agent11'])
    assert done.get_all_done()


    # Test with sim not done
    agents = {
        f'agent{n}': GridWorldAgent(id=f'agent{n}', encoding=n % 4 + 1) for n in range(12)
    }
    done = TargetEncodingInactiveDone(
        grid=grid,
        agents=agents,
        target_mapping={2: 1, 3: 1, 4:2},
        sim_ends_if_one_done=False
    )
    assert not done.sim_ends_if_one_done

    state.reset()
    for agent in agents.values():
        assert not done.get_done(agent)
    assert not done.get_all_done()

    agents['agent0'].active = False
    agents['agent4'].active = False
    agents['agent8'].active = False
    assert not done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert not done.get_done(agents['agent4'])
    assert done.get_done(agents['agent5'])
    assert done.get_done(agents['agent6'])
    assert not done.get_done(agents['agent7'])
    assert not done.get_done(agents['agent8'])
    assert done.get_done(agents['agent9'])
    assert done.get_done(agents['agent10'])
    assert not done.get_done(agents['agent11'])
    assert not done.get_all_done()

    agents['agent1'].active = False
    agents['agent5'].active = False
    agents['agent9'].active = False
    assert done.get_done(agents['agent3'])
    assert done.get_done(agents['agent7'])
    assert done.get_done(agents['agent11'])
    assert done.get_all_done()


    # Test agents targeting two encodings
    agents = {
        f'agent{n}': GridWorldAgent(id=f'agent{n}', encoding=n % 4 + 1) for n in range(12)
    }
    done = TargetEncodingInactiveDone(
        grid=grid,
        agents=agents,
        target_mapping={2: {1, 3}, 3: 1, 4:2},
    )

    state.reset()
    for agent in agents.values():
        assert not done.get_done(agent)
    assert not done.get_all_done()

    agents['agent0'].active = False
    agents['agent4'].active = False
    agents['agent8'].active = False
    assert not done.get_done(agents['agent0'])
    assert not done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert not done.get_done(agents['agent4'])
    assert not done.get_done(agents['agent5'])
    assert done.get_done(agents['agent6'])
    assert not done.get_done(agents['agent7'])
    assert not done.get_done(agents['agent8'])
    assert not done.get_done(agents['agent9'])
    assert done.get_done(agents['agent10'])
    assert not done.get_done(agents['agent11'])
    assert done.get_all_done()


    # Test agents targeting two encodings with sim not done
    agents = {
        f'agent{n}': GridWorldAgent(id=f'agent{n}', encoding=n % 4 + 1) for n in range(12)
    }
    done = TargetEncodingInactiveDone(
        grid=grid,
        agents=agents,
        target_mapping={2: {1, 3}, 3: 1, 4:2},
        sim_ends_if_one_done=False
    )

    state.reset()
    for agent in agents.values():
        assert not done.get_done(agent)
    assert not done.get_all_done()

    agents['agent0'].active = False
    agents['agent4'].active = False
    agents['agent8'].active = False
    assert not done.get_done(agents['agent0'])
    assert not done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert not done.get_done(agents['agent4'])
    assert not done.get_done(agents['agent5'])
    assert done.get_done(agents['agent6'])
    assert not done.get_done(agents['agent7'])
    assert not done.get_done(agents['agent8'])
    assert not done.get_done(agents['agent9'])
    assert done.get_done(agents['agent10'])
    assert not done.get_done(agents['agent11'])
    assert not done.get_all_done()


    agents['agent0'].active = False
    agents['agent2'].active = False
    agents['agent4'].active = False
    agents['agent6'].active = False
    agents['agent8'].active = False
    agents['agent10'].active = False
    assert not done.get_done(agents['agent0'])
    assert done.get_done(agents['agent1'])
    assert done.get_done(agents['agent2'])
    assert not done.get_done(agents['agent3'])
    assert not done.get_done(agents['agent4'])
    assert done.get_done(agents['agent5'])
    assert done.get_done(agents['agent6'])
    assert not done.get_done(agents['agent7'])
    assert not done.get_done(agents['agent8'])
    assert done.get_done(agents['agent9'])
    assert done.get_done(agents['agent10'])
    assert not done.get_done(agents['agent11'])
    assert not done.get_all_done()


    # Test failures
    with pytest.raises(AssertionError):
        TargetEncodingInactiveDone(
            grid=grid,
            agents=agents,
            target_mapping={2: 1, 3: 1, 4:2},
            sim_ends_if_one_done="True"
        )
    with pytest.raises(AssertionError):
        TargetEncodingInactiveDone(
            grid=grid,
            agents=agents,
            target_mapping={11: 1, 3: 1, 4:2},
        )
    with pytest.raises(AssertionError):
        TargetEncodingInactiveDone(
            grid=grid,
            agents=agents,
        )
    with pytest.raises(AssertionError):
        TargetEncodingInactiveDone(
            grid=grid,
            agents=agents,
        )
    with pytest.raises(AssertionError):
        TargetEncodingInactiveDone(
            grid=grid,
            agents=agents,
            target_mapping={
                5: [1, 2]
            }
        )
    with pytest.raises(TypeError):
        TargetEncodingInactiveDone(
            grid=grid,
            agents=agents,
            target_mapping={
                3: [1, 2]
            }
        )
    with pytest.raises(AssertionError):
        TargetEncodingInactiveDone(
            grid=grid,
            agents=agents,
            target_mapping={
                3: 3
            }
        )
    with pytest.raises(AssertionError):
        TargetEncodingInactiveDone(
            grid=grid,
            agents=agents,
            target_mapping={
                3: {3}
            }
        )
